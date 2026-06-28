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

import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import ladybug

from klit_flow.graph.model import GraphEdge, GraphNode

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    """Read a non-negative int from the environment, falling back to *default*."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


# Memory tuning for large repos. LadybugDB/Kuzu defaults ``buffer_pool_size`` to
# ~80% of *physical* RAM, which can OOM-kill (exit 137) the "Persisting graph …"
# step on a big app — especially since the parse/resolve results are still held
# in memory at that point. Cap the buffer pool and write parallelism by default;
# both are overridable via env vars for users who want to trade RAM for speed.
_DEFAULT_BUFFER_POOL_BYTES = 512 * 1024 * 1024  # 512 MiB
_DEFAULT_MAX_THREADS = 2

# Insert in bounded transactions: a single transaction over millions of rows
# would hold the whole change set in the WAL (defeating the point), while
# committing per row pays a checkpoint flush on every statement. Commit every
# ``_WRITE_BATCH_SIZE`` rows to bound peak memory without thrashing.
_WRITE_BATCH_SIZE = 5000

# A callback invoked with the running count of rows written, so callers can
# surface progress on the OOM-prone persist step (exit 137). Fired once per
# committed batch and once more at the end.
ProgressCallback = Callable[[int], None]


def parse_conditions_json(raw: str) -> list[dict[str, Any]]:
    """Parse a condition JSON string stored in the DB ``condition`` column.

    Returns a list of dicts with ``expression``, ``kind``, and optionally
    ``source_line``.  Gracefully returns an empty list for empty strings,
    ``None``, or malformed JSON.
    """
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return []


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
    def add_nodes(
        self, nodes: Iterable[GraphNode], *, progress: ProgressCallback | None = None
    ) -> None:
        """Upsert a batch of ``GraphNode`` records.

        *progress*, if given, is called with the running row count as batches
        commit, so callers can report progress on the persist step.
        """

    @abstractmethod
    def add_edges(
        self, edges: Iterable[GraphEdge], *, progress: ProgressCallback | None = None
    ) -> None:
        """Insert edges; silently skips any edge whose src or dst is absent.

        *progress*, if given, is called with the running row count as batches
        commit, so callers can report progress on the persist step.
        """

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

    def __init__(
        self,
        db_path: Path,
        emb_dim: int = 384,
        *,
        buffer_pool_size: int | None = None,
        max_num_threads: int | None = None,
    ) -> None:
        self._emb_dim = emb_dim
        self._lock = threading.Lock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if buffer_pool_size is None:
            buffer_pool_size = _env_int(
                "KLIT_FLOW_DB_BUFFER_POOL_BYTES", _DEFAULT_BUFFER_POOL_BYTES
            )
        if max_num_threads is None:
            max_num_threads = _env_int("KLIT_FLOW_DB_MAX_THREADS", _DEFAULT_MAX_THREADS)
        # ``buffer_pool_size=0`` / ``max_num_threads=0`` keep LadybugDB's auto
        # sizing (≈80% RAM, all cores); a positive value caps it.
        self._db = ladybug.Database(
            str(db_path),
            buffer_pool_size=buffer_pool_size,
            max_num_threads=max_num_threads,
        )
        self._conn = ladybug.Connection(self._db)
        self._closed = False
        self._checkpoint_supported = True
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
                trigger    STRING,
                condition  STRING
            )
        """)

    # ── Bulk-write helper ───────────────────────────────────────────────────────

    def _checkpoint(self) -> None:
        """Flush the WAL to the main store, best-effort.

        ``COMMIT`` only writes a batch to the WAL; the dirty pages and WAL
        entries are not reclaimed until a checkpoint moves them into the main
        store. Without this, memory climbs ~linearly with rows inserted and the
        OOM-prone persist step (exit 137) can be killed mid-write on a large
        repo. Run *outside* any active transaction (after ``COMMIT``).

        Failures are swallowed and disable further attempts: a missing/varying
        ``CHECKPOINT`` statement must never abort an otherwise-valid write.
        """
        if not self._checkpoint_supported:
            return
        try:
            self._conn.execute("CHECKPOINT").close()
        except Exception:
            self._checkpoint_supported = False
            logger.debug("CHECKPOINT unsupported; skipping further checkpoints.")

    def _batched_write(
        self,
        query: str,
        params_iter: Iterator[dict[str, Any]],
        *,
        progress: ProgressCallback | None = None,
    ) -> None:
        """Execute *query* once per param dict in bounded transactions.

        Each ``QueryResult`` is closed immediately so LadybugDB releases the
        result's native buffers instead of waiting on Python GC, a ``COMMIT``
        is issued every ``_WRITE_BATCH_SIZE`` rows so the in-memory WAL stays
        bounded on very large repos, and a ``CHECKPOINT`` after each commit
        flushes that WAL to disk so memory does not climb with the row count.
        The whole write rolls back on error.

        *progress* is called with the running row count after each committed
        batch (and once at the end), so callers can report how far the
        OOM-prone persist step has got before any crash.
        """
        rows = 0
        self._conn.execute("BEGIN TRANSACTION")
        try:
            for params in params_iter:
                self._conn.execute(query, params).close()
                rows += 1
                if rows % _WRITE_BATCH_SIZE == 0:
                    self._conn.execute("COMMIT")
                    self._checkpoint()
                    self._conn.execute("BEGIN TRANSACTION")
                    logger.info("  … %d rows written", rows)
                    if progress is not None:
                        progress(rows)
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        self._checkpoint()
        logger.info("  %d rows written (done)", rows)
        if progress is not None:
            progress(rows)

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def add_nodes(
        self, nodes: Iterable[GraphNode], *, progress: ProgressCallback | None = None
    ) -> None:
        """Upsert nodes via ``MERGE`` on the primary key."""
        query = (
            f"MERGE (n:{_NODE_TABLE} {{id: $id}}) "
            "ON CREATE SET n.kind = $kind, n.name = $name, "
            "n.file_path = $fp, n.start_line = $sl, n.end_line = $el, "
            "n.language = $lang "
            "ON MATCH SET n.kind = $kind, n.name = $name, "
            "n.file_path = $fp, n.start_line = $sl, n.end_line = $el, "
            "n.language = $lang"
        )
        params = (
            {
                "id": node.id,
                "kind": str(node.kind),
                "name": node.name,
                "fp": node.file_path,
                "sl": node.start_line,
                "el": node.end_line,
                "lang": node.language,
            }
            for node in nodes
        )
        self._batched_write(query, params, progress=progress)

    # ── Edges ─────────────────────────────────────────────────────────────────

    def add_edges(
        self, edges: Iterable[GraphEdge], *, progress: ProgressCallback | None = None
    ) -> None:
        """Insert edges.

        The ``MATCH`` pattern silently produces no result — and therefore
        creates no edge — when either endpoint is missing from the DB.
        """
        query = (
            f"MATCH (a:{_NODE_TABLE} {{id: $src}}), (b:{_NODE_TABLE} {{id: $dst}}) "
            f"CREATE (a)-[:{_EDGE_TABLE} {{type: $type, confidence: $conf, "
            f"file_path: $fp, line: $line, trigger: $trigger, condition: $cond}}]->(b)"
        )
        params = (
            {
                "src": edge.src_id,
                "dst": edge.dst_id,
                "type": str(edge.type),
                "conf": edge.confidence,
                "fp": edge.file_path or "",
                "line": edge.line or 0,
                "trigger": edge.trigger or "",
                "cond": json.dumps([c.model_dump() for c in edge.conditions])
                if edge.conditions
                else "",
            }
            for edge in edges
        )
        self._batched_write(query, params, progress=progress)

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, cypher: str) -> list[list[Any]]:
        """Execute *cypher* and return all rows as ``list[list[Any]]``."""
        with self._lock:
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
        with self._lock:
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
