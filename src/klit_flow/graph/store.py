"""Graph persistence layer.

``GraphStore`` ABC makes the storage engine swappable — callers outside
``store.py`` must only depend on the ABC, never on the concrete class.

``LadybugGraphStore`` is the concrete implementation backed by LadybugDB
(MIT fork of archived Kuzu v0.11.3; ``import ladybug``; API mirrors Kuzu
v0.11.3).

Schema
------
All seven node kinds (File, Module, Function, Class, Method, Interface,
Screen) share a single ``KlitNode`` node table, distinguished by the
``kind`` string column.  All six edge kinds share a single ``KlitEdge``
rel table, distinguished by the ``type`` column.  This keeps DDL minimal
and lets Cypher filters like ``WHERE kind = 'Screen'`` or
``WHERE type = 'NAVIGATES_TO'`` work without JOIN-like gymnastics.

The ``embedding FLOAT[emb_dim]`` column is populated by Phase 7; it is
NULL for nodes that have not yet been embedded.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import ladybug

from klit_flow.graph.model import GraphEdge, GraphNode

logger = logging.getLogger(__name__)

# Stable identifiers — never rename after first use; DB schema depends on them.
_NODE_TABLE = "KlitNode"
_EDGE_TABLE = "KlitEdge"
_VEC_INDEX = "klit_emb_idx"


class GraphStore(ABC):
    """Abstract persistence interface for the klit-flow knowledge graph.

    All callers outside ``store.py`` must program against this ABC so that
    swapping the storage engine requires changes only inside this module.
    """

    @abstractmethod
    def create_schema(self) -> None:
        """Create node / edge tables.  Must be idempotent (safe to call twice)."""

    @abstractmethod
    def add_nodes(self, nodes: list[GraphNode]) -> None:
        """Upsert a batch of ``GraphNode`` records."""

    @abstractmethod
    def add_edges(self, edges: list[GraphEdge]) -> None:
        """Insert edges; silently skips any edge whose src or dst is absent."""

    @abstractmethod
    def query(self, cypher: str) -> list[list[Any]]:
        """Execute a raw Cypher string and return all rows as ``list[list[Any]]``."""

    @abstractmethod
    def upsert_vectors(self, node_ids: list[str], embeddings: list[list[float]]) -> None:
        """Store embedding vectors on the given nodes and rebuild the vector index."""

    @abstractmethod
    def vector_search(self, embedding: list[float], k: int = 10) -> list[str]:
        """Return up to *k* node IDs ordered by cosine similarity to *embedding*."""

    @abstractmethod
    def close(self) -> None:
        """Flush writes and release the database connection."""

    def __enter__(self) -> GraphStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class LadybugGraphStore(GraphStore):
    """LadybugDB-backed ``GraphStore``.

    Parameters
    ----------
    db_path:
        Path where LadybugDB will create its storage directory.
        The *parent* directory must already exist; LadybugDB creates
        ``db_path`` itself.
    emb_dim:
        Dimensionality of embedding vectors.  Must match the embedding
        model used in Phase 7.  Defaults to 384 (``BAAI/bge-small-en-v1.5``).
    """

    def __init__(self, db_path: Path, emb_dim: int = 384) -> None:
        self._emb_dim = emb_dim
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = ladybug.Database(str(db_path))
        self._conn = ladybug.Connection(self._db)
        self._closed = False
        try:
            self._conn.execute("INSTALL VECTOR; LOAD EXTENSION VECTOR;")
        except Exception:
            logger.debug("Vector extension unavailable; vector_search will raise.")

    # ── Schema ────────────────────────────────────────────────────────────────

    def create_schema(self) -> None:
        """Create ``KlitNode`` and ``KlitEdge`` tables (idempotent)."""
        self._conn.execute(f"""
            CREATE NODE TABLE IF NOT EXISTS {_NODE_TABLE} (
                id         STRING,
                kind       STRING,
                name       STRING,
                file_path  STRING,
                start_line INT64,
                end_line   INT64,
                language   STRING,
                embedding  FLOAT[{self._emb_dim}],
                PRIMARY KEY (id)
            )
        """)
        self._conn.execute(f"""
            CREATE REL TABLE IF NOT EXISTS {_EDGE_TABLE} (
                FROM {_NODE_TABLE} TO {_NODE_TABLE},
                type       STRING,
                confidence DOUBLE,
                file_path  STRING,
                line       INT64,
                trigger    STRING
            )
        """)

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def add_nodes(self, nodes: list[GraphNode]) -> None:
        """Upsert nodes via ``MERGE`` on the primary key."""
        for node in nodes:
            self._conn.execute(
                f"MERGE (n:{_NODE_TABLE} {{id: $id}}) "
                "ON CREATE SET n.kind = $kind, n.name = $name, "
                "n.file_path = $fp, n.start_line = $sl, n.end_line = $el, "
                "n.language = $lang "
                "ON MATCH SET n.kind = $kind, n.name = $name, "
                "n.file_path = $fp, n.start_line = $sl, n.end_line = $el, "
                "n.language = $lang",
                {
                    "id": node.id,
                    "kind": str(node.kind),
                    "name": node.name,
                    "fp": node.file_path,
                    "sl": node.start_line,
                    "el": node.end_line,
                    "lang": node.language,
                },
            )

    # ── Edges ─────────────────────────────────────────────────────────────────

    def add_edges(self, edges: list[GraphEdge]) -> None:
        """Insert edges.

        The ``MATCH`` pattern silently produces no result — and therefore
        creates no edge — when either endpoint is missing from the DB.
        """
        for edge in edges:
            self._conn.execute(
                f"MATCH (a:{_NODE_TABLE} {{id: $src}}), (b:{_NODE_TABLE} {{id: $dst}}) "
                f"CREATE (a)-[:{_EDGE_TABLE} {{type: $type, confidence: $conf, "
                f"file_path: $fp, line: $line, trigger: $trigger}}]->(b)",
                {
                    "src": edge.src_id,
                    "dst": edge.dst_id,
                    "type": str(edge.type),
                    "conf": edge.confidence,
                    "fp": edge.file_path or "",
                    "line": edge.line or 0,
                    "trigger": edge.trigger or "",
                },
            )

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, cypher: str) -> list[list[Any]]:
        """Execute *cypher* and return all rows as ``list[list[Any]]``."""
        return self._conn.execute(cypher).get_all()

    # ── Vectors ───────────────────────────────────────────────────────────────

    def upsert_vectors(self, node_ids: list[str], embeddings: list[list[float]]) -> None:
        """Write embedding values and rebuild the cosine-similarity index."""
        for nid, vec in zip(node_ids, embeddings):
            vec_literal = "[" + ", ".join(f"{v:.8f}" for v in vec) + "]"
            self._conn.execute(
                f"MATCH (n:{_NODE_TABLE} {{id: $id}}) SET n.embedding = {vec_literal}",
                {"id": nid},
            )
        # CREATE_VECTOR_INDEX errors if the index already exists; ignore.
        try:
            self._conn.execute(
                f"CALL CREATE_VECTOR_INDEX('{_NODE_TABLE}', '{_VEC_INDEX}', "
                f"'embedding', metric := 'cosine')"
            )
        except Exception:
            pass

    def vector_search(self, embedding: list[float], k: int = 10) -> list[str]:
        """Return up to *k* node IDs nearest to *embedding* by cosine similarity."""
        vec_literal = "[" + ", ".join(f"{v:.8f}" for v in embedding) + "]"
        res = self._conn.execute(
            f"CALL QUERY_VECTOR_INDEX('{_NODE_TABLE}', '{_VEC_INDEX}', "
            f"{vec_literal}, {k}) RETURN node.id"
        )
        return [row[0] for row in res.get_all()]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the connection and database (idempotent)."""
        if not self._closed:
            self._conn.close()
            self._db.close()
            self._closed = True
