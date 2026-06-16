"""Phase 8 acceptance tests: MCP server smoke tests.

Each tool is exercised against a pre-indexed in-memory fixture.
Tests use ``asyncio.run(mcp.call_tool(...))`` — no stdio transport is
started; tools run purely in-process.

The ``_FakeEmbedder`` from Phase 7 is reused so no PyTorch / model
download is needed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from klit_flow.graph.model import GraphEdge, GraphNode, NodeKind, RelationType
from klit_flow.graph.store import LadybugGraphStore
from klit_flow.index.search import build_index
from klit_flow.server.mcp_server import create_server

# ---------------------------------------------------------------------------
# Fake embedder (no sentence-transformers / PyTorch needed)
# ---------------------------------------------------------------------------

_EMB: dict[str, list[float]] = {
    "Screen Auth Activity /src/AuthActivity.kt": [1.0, 0.0, 0.0, 0.0],
    "Screen Main Activity /src/MainActivity.kt": [0.0, 0.5, 0.5, 0.0],
    "Class Utils /src/Utils.kt": [0.0, 1.0, 0.0, 0.0],
    "auth": [0.95, 0.05, 0.0, 0.0],
}
_DEFAULT = [0.25, 0.25, 0.25, 0.25]


class _FakeEmbedder:
    dim = 4

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [_EMB.get(t, _DEFAULT) for t in texts]


# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

_AUTH = GraphNode(
    id="s_auth",
    kind=NodeKind.Screen,
    name="AuthActivity",
    file_path="/src/AuthActivity.kt",
    start_line=1,
    end_line=80,
    language="kotlin",
)
_MAIN = GraphNode(
    id="s_main",
    kind=NodeKind.Screen,
    name="MainActivity",
    file_path="/src/MainActivity.kt",
    start_line=1,
    end_line=100,
    language="kotlin",
)
_UTILS = GraphNode(
    id="c_utils",
    kind=NodeKind.Class,
    name="Utils",
    file_path="/src/Utils.kt",
    start_line=1,
    end_line=30,
    language="kotlin",
)

_NODES = [_AUTH, _MAIN, _UTILS]

_EDGES = [
    # Main imports Utils
    GraphEdge(src_id="s_main", dst_id="c_utils", type=RelationType.IMPORTS, confidence=1.0),
    # Main navigates to Auth
    GraphEdge(
        src_id="s_main",
        dst_id="s_auth",
        type=RelationType.NAVIGATES_TO,
        confidence=0.95,
        trigger="button_tap",
    ),
]


def _call(mcp, tool_name: str, args: dict) -> dict:
    """Synchronously invoke an MCP tool and parse the JSON result."""
    content, _ = asyncio.run(mcp.call_tool(tool_name, args))
    return json.loads(content[0].text)


# ---------------------------------------------------------------------------
# Server + store fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def mcp_fixture(tmp_path: Path):
    store = LadybugGraphStore(tmp_path / "graph.db", emb_dim=4)
    store.create_schema()
    store.add_nodes(_NODES)
    store.add_edges(_EDGES)
    bm25 = build_index(_NODES, store, _FakeEmbedder())
    mcp = create_server(store, bm25, _FakeEmbedder())
    yield mcp, store
    store.close()


# ---------------------------------------------------------------------------
# Tool: query
# ---------------------------------------------------------------------------


def test_query_returns_results(mcp_fixture) -> None:
    mcp, _ = mcp_fixture
    result = _call(mcp, "query", {"text": "auth"})
    assert "results" in result
    assert len(result["results"]) > 0


def test_query_auth_ranks_auth_activity_first(mcp_fixture) -> None:
    mcp, _ = mcp_fixture
    result = _call(mcp, "query", {"text": "auth"})
    ids = [r["id"] for r in result["results"]]
    assert "s_auth" in ids
    assert ids.index("s_auth") == 0


def test_query_result_has_required_fields(mcp_fixture) -> None:
    mcp, _ = mcp_fixture
    result = _call(mcp, "query", {"text": "auth"})
    for item in result["results"]:
        assert "id" in item
        assert "name" in item
        assert "kind" in item
        assert "file_path" in item


def test_query_empty_text_returns_list(mcp_fixture) -> None:
    mcp, _ = mcp_fixture
    result = _call(mcp, "query", {"text": "xyzzy_nonexistent"})
    assert "results" in result
    assert isinstance(result["results"], list)


# ---------------------------------------------------------------------------
# Tool: context
# ---------------------------------------------------------------------------


def test_context_known_symbol(mcp_fixture) -> None:
    mcp, _ = mcp_fixture
    result = _call(mcp, "context", {"symbol": "MainActivity"})
    assert result["node"] is not None
    assert result["node"]["name"] == "MainActivity"


def test_context_outbound_edges(mcp_fixture) -> None:
    """MainActivity imports Utils and navigates to Auth — both should appear."""
    mcp, _ = mcp_fixture
    result = _call(mcp, "context", {"symbol": "MainActivity"})
    out_types = {e["type"] for e in result["outbound"]}
    assert "IMPORTS" in out_types
    assert "NAVIGATES_TO" in out_types


def test_context_inbound_edges(mcp_fixture) -> None:
    """Utils is imported by Main — should appear as inbound."""
    mcp, _ = mcp_fixture
    result = _call(mcp, "context", {"symbol": "Utils"})
    in_names = {e["src_name"] for e in result["inbound"]}
    assert "MainActivity" in in_names


def test_context_unknown_symbol_returns_null_node(mcp_fixture) -> None:
    mcp, _ = mcp_fixture
    result = _call(mcp, "context", {"symbol": "NonExistentXYZ"})
    assert result["node"] is None
    assert result["outbound"] == []
    assert result["inbound"] == []


def test_context_result_shape(mcp_fixture) -> None:
    mcp, _ = mcp_fixture
    result = _call(mcp, "context", {"symbol": "MainActivity"})
    assert "node" in result
    assert "outbound" in result
    assert "inbound" in result


# ---------------------------------------------------------------------------
# Tool: impact
# ---------------------------------------------------------------------------


def test_impact_finds_importer(mcp_fixture) -> None:
    """Utils is imported by Main — Main should appear as a dependent."""
    mcp, _ = mcp_fixture
    result = _call(mcp, "impact", {"symbol": "Utils"})
    names = {d["name"] for d in result["dependents"]}
    assert "MainActivity" in names


def test_impact_no_dependents(mcp_fixture) -> None:
    """AuthActivity is not imported by anyone — dependents should be empty."""
    mcp, _ = mcp_fixture
    result = _call(mcp, "impact", {"symbol": "AuthActivity"})
    assert isinstance(result["dependents"], list)
    # AuthActivity has no importers/callers in the fixture
    assert result["dependents"] == []


def test_impact_result_shape(mcp_fixture) -> None:
    mcp, _ = mcp_fixture
    result = _call(mcp, "impact", {"symbol": "Utils"})
    assert "symbol" in result
    assert result["symbol"] == "Utils"
    assert "dependents" in result
    for d in result["dependents"]:
        assert "id" in d
        assert "name" in d
        assert "kind" in d


# ---------------------------------------------------------------------------
# Tool: flows
# ---------------------------------------------------------------------------


def test_flows_all_returns_navigates_to_edges(mcp_fixture) -> None:
    mcp, _ = mcp_fixture
    result = _call(mcp, "flows", {"screen": ""})
    assert "edges" in result
    assert len(result["edges"]) >= 1


def test_flows_all_edge_has_required_fields(mcp_fixture) -> None:
    mcp, _ = mcp_fixture
    result = _call(mcp, "flows", {"screen": ""})
    for edge in result["edges"]:
        assert "src_id" in edge
        assert "src_name" in edge
        assert "dst_id" in edge
        assert "dst_name" in edge
        assert "trigger" in edge
        assert "confidence" in edge


def test_flows_filtered_by_screen(mcp_fixture) -> None:
    mcp, _ = mcp_fixture
    result = _call(mcp, "flows", {"screen": "MainActivity"})
    # Every returned edge must involve MainActivity
    for edge in result["edges"]:
        assert edge["src_name"] == "MainActivity" or edge["dst_name"] == "MainActivity"


def test_flows_button_tap_trigger_preserved(mcp_fixture) -> None:
    mcp, _ = mcp_fixture
    result = _call(mcp, "flows", {"screen": ""})
    triggers = {e["trigger"] for e in result["edges"]}
    assert "button_tap" in triggers


def test_flows_unknown_screen_returns_empty(mcp_fixture) -> None:
    mcp, _ = mcp_fixture
    result = _call(mcp, "flows", {"screen": "NonExistentScreen"})
    assert result["edges"] == []


# ---------------------------------------------------------------------------
# Tool: cypher
# ---------------------------------------------------------------------------


def test_cypher_basic_match(mcp_fixture) -> None:
    mcp, _ = mcp_fixture
    result = _call(mcp, "cypher", {"q": "MATCH (n:KlitNode) RETURN n.id, n.name"})
    assert "rows" in result
    assert "columns" in result
    assert len(result["rows"]) == len(_NODES)


def test_cypher_columns_present(mcp_fixture) -> None:
    mcp, _ = mcp_fixture
    result = _call(mcp, "cypher", {"q": "MATCH (n:KlitNode) RETURN n.id"})
    assert "n.id" in result["columns"]


def test_cypher_filtered_query(mcp_fixture) -> None:
    mcp, _ = mcp_fixture
    result = _call(
        mcp,
        "cypher",
        {"q": "MATCH (n:KlitNode) WHERE n.kind = 'Screen' RETURN n.name ORDER BY n.name"},
    )
    names = [r[0] for r in result["rows"]]
    assert "AuthActivity" in names
    assert "MainActivity" in names
    assert "Utils" not in names


def test_cypher_empty_result(mcp_fixture) -> None:
    mcp, _ = mcp_fixture
    result = _call(
        mcp,
        "cypher",
        {"q": "MATCH (n:KlitNode) WHERE n.name = 'DoesNotExist' RETURN n.id"},
    )
    assert result["rows"] == []


# ---------------------------------------------------------------------------
# Server registration
# ---------------------------------------------------------------------------


def test_server_has_five_tools(mcp_fixture) -> None:
    mcp, _ = mcp_fixture
    tools = asyncio.run(mcp.list_tools())
    tool_names = {t.name for t in tools}
    assert tool_names == {"query", "context", "impact", "flows", "cypher"}


def test_server_name_is_klit_flow(mcp_fixture) -> None:
    mcp, _ = mcp_fixture
    assert mcp.name == "klit-flow"
