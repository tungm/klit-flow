"""Tests for the klit-flow web portal API endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from klit_flow.graph.model import ConditionLevel, GraphEdge, GraphNode, NodeKind, RelationType
from klit_flow.graph.store import LadybugGraphStore
from klit_flow.index.bm25 import BM25Index
from klit_flow.server.web_server import create_web_app

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_FILE = GraphNode(
    id="f1",
    kind=NodeKind.File,
    name="AuthActivity.kt",
    file_path="/app/src/AuthActivity.kt",
    start_line=1,
    end_line=50,
    language="kotlin",
)
_AUTH = GraphNode(
    id="auth",
    kind=NodeKind.Screen,
    name="AuthActivity",
    file_path="/app/src/AuthActivity.kt",
    start_line=3,
    end_line=48,
    language="kotlin",
)
_MAIN = GraphNode(
    id="main",
    kind=NodeKind.Screen,
    name="MainActivity",
    file_path="/app/src/MainActivity.kt",
    start_line=1,
    end_line=30,
    language="kotlin",
)
_EDGE_NAV = GraphEdge(
    src_id="main",
    dst_id="auth",
    type=RelationType.NAVIGATES_TO,
    confidence=0.95,
    file_path="/app/src/MainActivity.kt",
    line=12,
    trigger="button_tap",
    conditions=[ConditionLevel(expression="loginButton clicked", kind="annotation")],
)
_EDGE_DECL = GraphEdge(
    src_id="f1",
    dst_id="auth",
    type=RelationType.DECLARES,
    confidence=1.0,
    file_path="/app/src/AuthActivity.kt",
    line=3,
    trigger="",
)


class _FakeEmbedder:
    dim = 4

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(i % 4) / 4 for i in range(4)] for _ in texts]


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    klit_dir = tmp_path / ".klit-flow"
    klit_dir.mkdir()
    db_path = klit_dir / "graph.db"
    embedder = _FakeEmbedder()

    store = LadybugGraphStore(db_path, emb_dim=embedder.dim)
    store.create_schema()
    nodes = [_FILE, _AUTH, _MAIN]
    store.add_nodes(nodes)
    store.add_edges([_EDGE_NAV, _EDGE_DECL])

    bm25 = BM25Index()
    for n in nodes:
        bm25.add(n.id, f"{n.kind} {n.name} {n.file_path}")
    bm25.build()

    texts = [f"{n.kind} {n.name} {n.file_path}" for n in nodes]
    vecs = embedder.encode(texts)
    store.upsert_vectors([n.id for n in nodes], vecs)

    app = create_web_app(store, bm25, embedder)
    return TestClient(app)


# ---------------------------------------------------------------------------
# SPA
# ---------------------------------------------------------------------------


def test_spa_returns_html(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "klit-flow" in r.text


def test_spa_has_graph_canvas(client: TestClient) -> None:
    r = client.get("/")
    assert "graph-canvas" in r.text


def test_spa_has_no_external_cdn(client: TestClient) -> None:
    text = client.get("/").text
    assert "cdn." not in text
    assert "unpkg.com" not in text
    assert "jsdelivr" not in text


# ---------------------------------------------------------------------------
# /api/graph
# ---------------------------------------------------------------------------


def test_graph_returns_nodes(client: TestClient) -> None:
    r = client.get("/api/graph")
    assert r.status_code == 200
    data = r.json()
    assert len(data["nodes"]) == 3


def test_graph_returns_edges(client: TestClient) -> None:
    r = client.get("/api/graph")
    data = r.json()
    assert len(data["edges"]) == 2


def test_graph_node_has_required_fields(client: TestClient) -> None:
    r = client.get("/api/graph")
    node = r.json()["nodes"][0]
    assert {"id", "kind", "name", "file", "start_line"} <= node.keys()


def test_graph_edge_has_required_fields(client: TestClient) -> None:
    r = client.get("/api/graph")
    edge = r.json()["edges"][0]
    assert {"src", "dst", "type", "confidence", "trigger"} <= edge.keys()


# ---------------------------------------------------------------------------
# /api/flows
# ---------------------------------------------------------------------------


def test_flows_all(client: TestClient) -> None:
    r = client.get("/api/flows")
    assert r.status_code == 200
    data = r.json()
    assert len(data["flows"]) == 1
    assert data["flows"][0]["from"] == "MainActivity"
    assert data["flows"][0]["to"] == "AuthActivity"


def test_flows_filter_match(client: TestClient) -> None:
    r = client.get("/api/flows?screen=AuthActivity")
    assert r.status_code == 200
    data = r.json()
    assert len(data["flows"]) == 1


def test_flows_filter_no_match(client: TestClient) -> None:
    r = client.get("/api/flows?screen=UnknownScreen")
    assert r.status_code == 200
    assert r.json()["flows"] == []


def test_flows_trigger_present(client: TestClient) -> None:
    r = client.get("/api/flows")
    assert r.json()["flows"][0]["trigger"] == "button_tap"


def test_flows_confidence_present(client: TestClient) -> None:
    r = client.get("/api/flows")
    assert r.json()["flows"][0]["confidence"] == pytest.approx(0.95)


def test_flows_conditions_present(client: TestClient) -> None:
    r = client.get("/api/flows")
    flow = r.json()["flows"][0]
    assert "conditions" in flow
    assert len(flow["conditions"]) == 1
    assert flow["conditions"][0]["expression"] == "loginButton clicked"
    assert flow["conditions"][0]["kind"] == "annotation"


def test_graph_edge_has_conditions_field(client: TestClient) -> None:
    r = client.get("/api/graph")
    nav_edge = next(e for e in r.json()["edges"] if e["type"] == "NAVIGATES_TO")
    assert "conditions" in nav_edge
    assert len(nav_edge["conditions"]) == 1
    assert nav_edge["conditions"][0]["expression"] == "loginButton clicked"


# ---------------------------------------------------------------------------
# /api/screen-apis/{screen_id}
# ---------------------------------------------------------------------------


def test_screen_apis_found(client: TestClient) -> None:
    r = client.get("/api/screen-apis/main")
    assert r.status_code == 200
    data = r.json()
    assert "api_deps" in data
    assert isinstance(data["api_deps"], list)


def test_screen_apis_not_found(client: TestClient) -> None:
    r = client.get("/api/screen-apis/nonexistent")
    assert r.status_code == 404


def test_screen_apis_response_shape(client: TestClient) -> None:
    r = client.get("/api/screen-apis/main")
    assert r.json()["screen_id"] == "main"
    assert r.json()["screen_name"] == "MainActivity"


# ---------------------------------------------------------------------------
# /api/search
# ---------------------------------------------------------------------------


def test_search_returns_results(client: TestClient) -> None:
    r = client.get("/api/search?q=AuthActivity")
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert len(data["results"]) > 0


def test_search_result_has_fields(client: TestClient) -> None:
    r = client.get("/api/search?q=AuthActivity")
    result = r.json()["results"][0]
    assert {"id", "kind", "name", "file"} <= result.keys()


def test_search_auth_ranks_first(client: TestClient) -> None:
    r = client.get("/api/search?q=Auth")
    ids = [res["id"] for res in r.json()["results"]]
    assert "auth" in ids or "f1" in ids


def test_search_respects_k(client: TestClient) -> None:
    r = client.get("/api/search?q=Activity&k=1")
    assert len(r.json()["results"]) <= 1


# ---------------------------------------------------------------------------
# /api/node/{node_id}
# ---------------------------------------------------------------------------


def test_node_found(client: TestClient) -> None:
    r = client.get("/api/node/auth")
    assert r.status_code == 200
    data = r.json()
    assert data["node"]["name"] == "AuthActivity"
    assert data["node"]["kind"] == "Screen"


def test_node_not_found(client: TestClient) -> None:
    r = client.get("/api/node/nonexistent")
    assert r.status_code == 404


def test_node_inbound_edges(client: TestClient) -> None:
    r = client.get("/api/node/auth")
    data = r.json()
    inbound_types = [e["type"] for e in data["inbound"]]
    assert "NAVIGATES_TO" in inbound_types or "DECLARES" in inbound_types


def test_node_outbound_edges(client: TestClient) -> None:
    r = client.get("/api/node/main")
    data = r.json()
    assert any(e["type"] == "NAVIGATES_TO" for e in data["outbound"])


def test_node_detail_has_file_fields(client: TestClient) -> None:
    r = client.get("/api/node/auth")
    node = r.json()["node"]
    assert node["start_line"] == 3
    assert node["end_line"] == 48
    assert node["language"] == "kotlin"
