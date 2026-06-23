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


# ---------------------------------------------------------------------------
# /api/named-flows
# ---------------------------------------------------------------------------

# The fixture graph has exactly one NAVIGATES_TO edge: MainActivity -> AuthActivity.
_VALID = {
    "name": "Login",
    "branches": [
        {
            "screens": [
                {"id": "main", "name": "MainActivity"},
                {"id": "auth", "name": "AuthActivity"},
            ]
        }
    ],
}


def test_named_flow_create_valid(client: TestClient) -> None:
    r = client.post("/api/named-flows", json=_VALID)
    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "Login"
    assert len(body["branches"]) == 1
    assert [s["name"] for s in body["branches"][0]["screens"]] == [
        "MainActivity",
        "AuthActivity",
    ]
    assert body["id"]


def test_named_flow_create_rejects_unconnected_screens(client: TestClient) -> None:
    # auth -> main is NOT a navigation edge in the fixture graph.
    bad = {
        "name": "Backwards",
        "branches": [
            {
                "screens": [
                    {"id": "auth", "name": "AuthActivity"},
                    {"id": "main", "name": "MainActivity"},
                ]
            }
        ],
    }
    r = client.post("/api/named-flows", json=bad)
    assert r.status_code == 400
    assert "navigation edge" in r.json()["detail"]


def test_named_flow_create_allows_single_screen_branch(client: TestClient) -> None:
    r = client.post(
        "/api/named-flows",
        json={"name": "Solo", "branches": [{"screens": [{"id": "main", "name": "MainActivity"}]}]},
    )
    assert r.status_code == 201
    assert [s["name"] for s in r.json()["branches"][0]["screens"]] == ["MainActivity"]


def test_named_flow_create_rejects_empty_branch(client: TestClient) -> None:
    r = client.post(
        "/api/named-flows",
        json={"name": "Empty branch", "branches": [{"screens": []}]},
    )
    assert r.status_code == 400


def test_named_flow_create_requires_a_branch(client: TestClient) -> None:
    r = client.post("/api/named-flows", json={"name": "Empty", "branches": []})
    assert r.status_code == 400


def test_named_flow_create_requires_name(client: TestClient) -> None:
    r = client.post("/api/named-flows", json={"name": "  ", "branches": _VALID["branches"]})
    assert r.status_code == 400


def test_named_flow_list_returns_created(client: TestClient) -> None:
    client.post("/api/named-flows", json=_VALID)
    r = client.get("/api/named-flows")
    assert r.status_code == 200
    assert len(r.json()["flows"]) == 1


def test_named_flow_search_subsequence(client: TestClient) -> None:
    client.post("/api/named-flows", json=_VALID)
    # Single screen is an ordered subsequence of the flow.
    r = client.get("/api/named-flows?q=MainActivity")
    assert len(r.json()["flows"]) == 1
    # Full sequence matches.
    r = client.get("/api/named-flows?q=MainActivity > AuthActivity")
    assert len(r.json()["flows"]) == 1
    # Reversed order does not match.
    r = client.get("/api/named-flows?q=AuthActivity > MainActivity")
    assert len(r.json()["flows"]) == 0


def test_named_flow_search_case_insensitive(client: TestClient) -> None:
    client.post("/api/named-flows", json=_VALID)
    r = client.get("/api/named-flows?q=mainactivity")
    assert len(r.json()["flows"]) == 1


@pytest.mark.parametrize(
    "q",
    [
        "MainActivity > AuthActivity",
        "MainActivity -> AuthActivity",
        "MainActivity, AuthActivity",
    ],
)
def test_named_flow_search_accepts_each_separator(client: TestClient, q: str) -> None:
    client.post("/api/named-flows", json=_VALID)
    r = client.get("/api/named-flows", params={"q": q})
    assert len(r.json()["flows"]) == 1


def test_named_flow_rename(client: TestClient) -> None:
    fid = client.post("/api/named-flows", json=_VALID).json()["id"]
    r = client.put(f"/api/named-flows/{fid}", json={"name": "Auth flow"})
    assert r.status_code == 200
    assert r.json()["name"] == "Auth flow"


def test_named_flow_update_rejects_unconnected_screens(client: TestClient) -> None:
    fid = client.post("/api/named-flows", json=_VALID).json()["id"]
    r = client.put(
        f"/api/named-flows/{fid}",
        json={
            "branches": [
                {
                    "screens": [
                        {"id": "auth", "name": "AuthActivity"},
                        {"id": "main", "name": "MainActivity"},
                    ]
                }
            ]
        },
    )
    assert r.status_code == 400


def test_named_flow_delete(client: TestClient) -> None:
    fid = client.post("/api/named-flows", json=_VALID).json()["id"]
    assert client.delete(f"/api/named-flows/{fid}").status_code == 200
    assert client.get(f"/api/named-flows/{fid}").status_code == 404


def test_named_flow_get_not_found(client: TestClient) -> None:
    assert client.get("/api/named-flows/nope").status_code == 404


def test_spa_has_named_flows_tab(client: TestClient) -> None:
    assert 'data-view="named"' in client.get("/").text


def test_spa_has_tree_builder(client: TestClient) -> None:
    text = client.get("/").text
    assert 'id="nf-tree"' in text  # branch tree builder
    assert "nfTreeFromBranches" in text  # tree<->branches round-trip logic


def test_spa_has_export_import_controls(client: TestClient) -> None:
    text = client.get("/").text
    assert 'id="nf-export"' in text
    assert 'id="nf-import-file"' in text


# ---------------------------------------------------------------------------
# /api/named-flows/export  +  /api/named-flows/import
# ---------------------------------------------------------------------------


def test_named_flow_export_structure(client: TestClient) -> None:
    client.post("/api/named-flows", json=_VALID)
    r = client.get("/api/named-flows/export")
    assert r.status_code == 200  # 'export' must win over /{flow_id}
    data = r.json()
    assert data["version"] >= 1
    assert "exported_at" in data
    assert len(data["flows"]) == 1
    screen = data["flows"][0]["branches"][0]["screens"][0]
    # Each screen carries dependency + API enrichment (possibly empty lists).
    assert isinstance(screen["dependencies"], list)
    assert isinstance(screen["apis"], list)
    assert {"id", "name"} <= screen.keys()


def test_named_flow_export_empty(client: TestClient) -> None:
    data = client.get("/api/named-flows/export").json()
    assert data["flows"] == []


def test_named_flow_import_roundtrip(client: TestClient) -> None:
    client.post("/api/named-flows", json=_VALID)
    exported = client.get("/api/named-flows/export").json()
    # Re-importing appends a copy.
    r = client.post("/api/named-flows/import", json=exported)
    assert r.status_code == 200
    assert r.json()["imported"] == 1
    assert len(client.get("/api/named-flows").json()["flows"]) == 2


def test_named_flow_import_ignores_enrichment_and_skips_validation(client: TestClient) -> None:
    # auth -> main is NOT a nav edge, and screens carry enrichment fields.
    payload = {
        "flows": [
            {
                "name": "Imported",
                "branches": [
                    {
                        "screens": [
                            {"id": "auth", "name": "AuthActivity", "dependencies": [], "apis": []},
                            {"id": "main", "name": "MainActivity", "dependencies": [], "apis": []},
                        ]
                    }
                ],
            }
        ]
    }
    r = client.post("/api/named-flows/import", json=payload)
    assert r.status_code == 200
    assert r.json()["imported"] == 1
    flows = client.get("/api/named-flows").json()["flows"]
    assert any(f["name"] == "Imported" for f in flows)


def test_named_flow_import_skips_unnamed_or_empty(client: TestClient) -> None:
    payload = {
        "flows": [
            {"name": "  ", "branches": [{"screens": [{"id": "main", "name": "MainActivity"}]}]},
            {"name": "NoBranches", "branches": []},
        ]
    }
    r = client.post("/api/named-flows/import", json=payload)
    assert r.json()["imported"] == 0


def test_export_enriches_screen_with_deps_and_apis(tmp_path: Path) -> None:
    # Graph: HomeActivity (screen) whose class/file reach a Repository -> ApiService.
    embedder = _FakeEmbedder()
    store = LadybugGraphStore(tmp_path / "graph.db", emb_dim=embedder.dim)
    store.create_schema()
    nodes = [
        _screen("home", "HomeActivity"),  # file /app/src/HomeActivity.kt
        GraphNode(
            id="hc",
            kind=NodeKind.Class,
            name="HomeActivity",
            file_path="/app/src/HomeActivity.kt",
            start_line=1,
            end_line=20,
            language="kotlin",
        ),
        GraphNode(
            id="repo",
            kind=NodeKind.Class,
            name="UserRepository",
            file_path="/app/src/UserRepository.kt",
            start_line=1,
            end_line=20,
            language="kotlin",
        ),
        GraphNode(
            id="api",
            kind=NodeKind.Class,
            name="UserApiService",
            file_path="/app/src/UserApiService.kt",
            start_line=1,
            end_line=20,
            language="kotlin",
        ),
    ]
    store.add_nodes(nodes)
    store.add_edges(
        [
            GraphEdge(src_id="hc", dst_id="repo", type=RelationType.CALLS, confidence=0.8),
            GraphEdge(src_id="repo", dst_id="api", type=RelationType.CALLS, confidence=0.8),
        ]
    )
    bm25 = BM25Index()
    bm25.build()
    client = TestClient(create_web_app(store, bm25, embedder))

    client.post(
        "/api/named-flows",
        json={"name": "Home", "branches": [{"screens": [{"id": "home", "name": "HomeActivity"}]}]},
    )
    screen = client.get("/api/named-flows/export").json()["flows"][0]["branches"][0]["screens"][0]

    dep_names = {d["name"] for d in screen["dependencies"]}
    api_names = {a["name"] for a in screen["apis"]}
    assert "UserRepository" in dep_names  # direct 1-hop dependency
    assert {"UserRepository", "UserApiService"} <= api_names  # reachable API-ish nodes


# ---------------------------------------------------------------------------
# Branching named flows (richer graph: A -> B -> {C1 -> D, C2})
# ---------------------------------------------------------------------------


def _screen(node_id: str, name: str) -> GraphNode:
    return GraphNode(
        id=node_id,
        kind=NodeKind.Screen,
        name=name,
        file_path=f"/app/src/{name}.kt",
        start_line=1,
        end_line=10,
        language="kotlin",
    )


def _nav(src: str, dst: str) -> GraphEdge:
    return GraphEdge(src_id=src, dst_id=dst, type=RelationType.NAVIGATES_TO, confidence=0.9)


@pytest.fixture()
def branching_client(tmp_path: Path) -> TestClient:
    embedder = _FakeEmbedder()
    store = LadybugGraphStore(tmp_path / "graph.db", emb_dim=embedder.dim)
    store.create_schema()
    nodes = [
        _screen("a", "A"),
        _screen("b", "B"),
        _screen("c1", "C1"),
        _screen("c2", "C2"),
        _screen("d", "D"),
    ]
    store.add_nodes(nodes)
    store.add_edges([_nav("a", "b"), _nav("b", "c1"), _nav("c1", "d"), _nav("b", "c2")])
    bm25 = BM25Index()
    bm25.build()
    return TestClient(create_web_app(store, bm25, embedder))


_TWO_BRANCH = {
    "name": "Checkout",
    "branches": [
        {
            "screens": [
                {"id": "a", "name": "A"},
                {"id": "b", "name": "B"},
                {"id": "c1", "name": "C1"},
                {"id": "d", "name": "D"},
            ]
        },
        {
            "screens": [
                {"id": "a", "name": "A"},
                {"id": "b", "name": "B"},
                {"id": "c2", "name": "C2"},
            ]
        },
    ],
}


def test_branching_create_two_branches(branching_client: TestClient) -> None:
    r = branching_client.post("/api/named-flows", json=_TWO_BRANCH)
    assert r.status_code == 201
    assert len(r.json()["branches"]) == 2


def test_branching_rejects_invalid_branch(branching_client: TestClient) -> None:
    # Second branch has a non-edge: C1 -> C2 is not a NAVIGATES_TO edge.
    bad = {
        "name": "Bad",
        "branches": [
            {"screens": [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}]},
            {"screens": [{"id": "c1", "name": "C1"}, {"id": "c2", "name": "C2"}]},
        ],
    }
    r = branching_client.post("/api/named-flows", json=bad)
    assert r.status_code == 400


def test_branching_search_matches_either_branch(branching_client: TestClient) -> None:
    branching_client.post("/api/named-flows", json=_TWO_BRANCH)
    # In the first branch only.
    assert len(branching_client.get("/api/named-flows?q=C1 > D").json()["flows"]) == 1
    # In the second branch only.
    assert len(branching_client.get("/api/named-flows?q=B > C2").json()["flows"]) == 1
    # Shared prefix matches via either branch.
    assert len(branching_client.get("/api/named-flows?q=A > B").json()["flows"]) == 1


def test_branching_search_no_cross_branch_match(branching_client: TestClient) -> None:
    branching_client.post("/api/named-flows", json=_TWO_BRANCH)
    # C1 and C2 live in different branches, so a sequence spanning both must not match.
    assert len(branching_client.get("/api/named-flows?q=C1 > C2").json()["flows"]) == 0
