"""klit-flow MCP server.

Exposes five tools over the Model Context Protocol (stdio transport):

- ``query(text)``    — hybrid BM25 + semantic search; returns ranked node list.
- ``context(symbol)`` — inbound / outbound edges for a named symbol.
- ``impact(symbol)``  — direct importers and callers (reverse 1-hop traversal).
- ``flows(screen)``   — NAVIGATES_TO edges, optionally filtered to one screen.
- ``cypher(q)``       — raw Cypher passthrough (read-only by convention).

The server is created by :func:`create_server`, which accepts the already-open
:class:`~klit_flow.graph.store.GraphStore`, a :class:`~klit_flow.index.bm25.BM25Index`,
and an *embedder* object (anything with ``encode(texts) -> list[list[float]]``).

The :func:`main` entry point is wired to ``klit-flow serve`` and loads these
from the ``.klit-flow/`` directory of the current working directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from klit_flow.graph.store import GraphStore, parse_conditions_json
from klit_flow.index.bm25 import BM25Index
from klit_flow.index.search import hybrid_search

if TYPE_CHECKING:
    from klit_flow.index.embeddings import Embedder

logger = logging.getLogger(__name__)

_SERVER_NAME = "klit-flow"


def _esc(s: str) -> str:
    """Escape a string for inline use in a Cypher literal."""
    return s.replace("\\", "\\\\").replace("'", "\\'")


def _rows_to_dicts(rows: list[list[Any]], keys: list[str]) -> list[dict[str, Any]]:
    return [dict(zip(keys, row)) for row in rows]


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def create_server(
    store: GraphStore,
    bm25: BM25Index,
    embedder: Embedder,
) -> FastMCP:
    """Build and return a :class:`FastMCP` instance with all five tools registered.

    Parameters
    ----------
    store:
        Open :class:`~klit_flow.graph.store.GraphStore` (must remain open for
        the server's lifetime).
    bm25:
        Built :class:`~klit_flow.index.bm25.BM25Index`.
    embedder:
        Any object with ``encode(texts: list[str]) -> list[list[float]]``.
        Pass a ``_FakeEmbedder`` in tests.
    """
    mcp = FastMCP(_SERVER_NAME)

    # ── query ─────────────────────────────────────────────────────────────────

    @mcp.tool()
    def query(text: str) -> str:
        """Hybrid BM25 + semantic search over all indexed graph nodes.

        Returns a JSON object with a ``results`` list; each item has
        ``id``, ``name``, ``kind``, and ``file_path``.
        """
        node_ids = hybrid_search(text, bm25, store, embedder, k=10)
        if not node_ids:
            return json.dumps({"results": []})

        id_list = ", ".join(f"'{_esc(nid)}'" for nid in node_ids)
        rows = store.query(
            f"MATCH (n:KlitNode) WHERE n.id IN [{id_list}] RETURN n.id, n.name, n.kind, n.file_path"
        )
        by_id = {r[0]: r for r in rows}
        results = []
        for nid in node_ids:  # preserve ranked order
            if nid in by_id:
                r = by_id[nid]
                results.append({"id": r[0], "name": r[1], "kind": r[2], "file_path": r[3]})

        return json.dumps({"results": results})

    # ── context ───────────────────────────────────────────────────────────────

    @mcp.tool()
    def context(symbol: str) -> str:
        """Return inbound and outbound edges for a node identified by name.

        Returns a JSON object with ``node``, ``outbound`` (edges leaving
        the node), and ``inbound`` (edges entering the node).  If no node
        with the given name exists, ``node`` is ``null``.
        """
        sym = _esc(symbol)

        node_rows = store.query(
            f"MATCH (n:KlitNode) WHERE n.name = '{sym}' "
            f"RETURN n.id, n.name, n.kind, n.file_path LIMIT 1"
        )
        if not node_rows:
            return json.dumps({"node": None, "outbound": [], "inbound": []})

        nid, name, kind, fp = node_rows[0]
        node = {"id": nid, "name": name, "kind": kind, "file_path": fp}

        out_rows = store.query(
            f"MATCH (n:KlitNode {{id: '{nid}'}})-[e:KlitEdge]->(m:KlitNode) "
            f"RETURN e.type, m.id, m.name, m.kind, e.trigger, e.confidence, e.condition"
        )
        outbound = [
            {
                "type": r[0],
                "dst_id": r[1],
                "dst_name": r[2],
                "dst_kind": r[3],
                "trigger": r[4],
                "confidence": r[5],
                "conditions": parse_conditions_json(r[6]),
            }
            for r in out_rows
        ]

        in_rows = store.query(
            f"MATCH (m:KlitNode)-[e:KlitEdge]->(n:KlitNode {{id: '{nid}'}}) "
            f"RETURN e.type, m.id, m.name, m.kind, e.trigger, e.confidence, e.condition"
        )
        inbound = [
            {
                "type": r[0],
                "src_id": r[1],
                "src_name": r[2],
                "src_kind": r[3],
                "trigger": r[4],
                "confidence": r[5],
                "conditions": parse_conditions_json(r[6]),
            }
            for r in in_rows
        ]

        return json.dumps({"node": node, "outbound": outbound, "inbound": inbound})

    # ── impact ────────────────────────────────────────────────────────────────

    @mcp.tool()
    def impact(symbol: str) -> str:
        """Find all nodes that directly import or call the given symbol.

        Uses a 1-hop reverse traversal over IMPORTS and CALLS edges.
        Returns ``{"symbol": ..., "dependents": [...]}`` where each
        dependent has ``id``, ``name``, and ``kind``.
        """
        sym = _esc(symbol)
        rows = store.query(
            f"MATCH (a:KlitNode)-[e:KlitEdge]->(b:KlitNode) "
            f"WHERE b.name = '{sym}' AND (e.type = 'IMPORTS' OR e.type = 'CALLS') "
            f"RETURN DISTINCT a.id, a.name, a.kind"
        )
        dependents = [{"id": r[0], "name": r[1], "kind": r[2]} for r in rows]
        return json.dumps({"symbol": symbol, "dependents": dependents})

    # ── flows ─────────────────────────────────────────────────────────────────

    @mcp.tool()
    def flows(screen: str = "") -> str:
        """Return NAVIGATES_TO edges, optionally scoped to one screen.

        If *screen* is empty, returns all navigation edges in the graph.
        If *screen* is set, returns edges where the source or destination
        screen name matches.

        Each edge has ``src_id``, ``src_name``, ``dst_id``, ``dst_name``,
        ``trigger``, ``confidence``, and ``conditions`` (structured list of
        condition levels with ``expression``, ``kind``, and ``source_line``).
        """
        if screen:
            sym = _esc(screen)
            rows = store.query(
                f"MATCH (a:KlitNode)-[e:KlitEdge]->(b:KlitNode) "
                f"WHERE e.type = 'NAVIGATES_TO' "
                f"AND (a.name = '{sym}' OR b.name = '{sym}') "
                f"RETURN a.id, a.name, b.id, b.name, e.trigger, e.confidence, e.condition"
            )
        else:
            rows = store.query(
                "MATCH (a:KlitNode)-[e:KlitEdge]->(b:KlitNode) "
                "WHERE e.type = 'NAVIGATES_TO' "
                "RETURN a.id, a.name, b.id, b.name, e.trigger, e.confidence, e.condition"
            )

        edges = [
            {
                "src_id": r[0],
                "src_name": r[1],
                "dst_id": r[2],
                "dst_name": r[3],
                "trigger": r[4],
                "confidence": r[5],
                "conditions": parse_conditions_json(r[6]),
            }
            for r in rows
        ]
        return json.dumps({"screen": screen or None, "edges": edges})

    # ── cypher ────────────────────────────────────────────────────────────────

    @mcp.tool()
    def cypher(q: str) -> str:
        """Execute a raw Cypher query and return all rows as JSON.

        Returns ``{"rows": [[...], ...], "columns": [...]}`` where each
        inner list corresponds to one result row.

        .. warning::
            This is a read-only tool by convention.  No DDL guards are
            enforced; avoid running mutating statements via this tool.
        """
        result = store._conn.execute(q)  # noqa: SLF001 — raw passthrough
        columns = result.get_column_names()
        rows = result.get_all()
        return json.dumps({"columns": columns, "rows": rows})

    return mcp


# ---------------------------------------------------------------------------
# CLI entry point (wired to ``klit-flow serve``)
# ---------------------------------------------------------------------------

_KLIT_DIR = ".klit-flow"
_DB_PATH = "graph.db"
_BM25_PATH = "bm25.pkl"


def main(target: Path | None = None) -> None:
    """Load index from ``.klit-flow/`` and start the MCP server over stdio.

    Parameters
    ----------
    target:
        Root directory of the target repo.  Defaults to ``Path.cwd()``.
    """
    from klit_flow.graph.store import LadybugGraphStore
    from klit_flow.index.embeddings import Embedder

    root = target or Path.cwd()
    klit_dir = root / _KLIT_DIR

    db_path = klit_dir / _DB_PATH
    bm25_path = klit_dir / _BM25_PATH

    if not db_path.exists():
        raise FileNotFoundError(f"No index found at {db_path}. Run 'klit-flow analyze' first.")

    store = LadybugGraphStore(db_path)
    bm25 = BM25Index.load(bm25_path) if bm25_path.exists() else _empty_bm25()
    embedder = Embedder()

    server = create_server(store, bm25, embedder)
    try:
        server.run(transport="stdio")
    finally:
        store.close()


def _empty_bm25() -> BM25Index:
    idx = BM25Index()
    idx.build()
    return idx
