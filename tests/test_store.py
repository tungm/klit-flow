"""Phase 6 acceptance tests: GraphStore persistence layer.

Round-trip: add nodes + edges, run Cypher queries, verify expected rows.
All tests run fully offline — LadybugDB is embedded, no network required.
"""

from pathlib import Path

import pytest

from klit_flow.graph.model import GraphEdge, GraphNode, NodeKind, RelationType
from klit_flow.graph.store import GraphStore, LadybugGraphStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_file_node(node_id: str, name: str, path: str) -> GraphNode:
    return GraphNode(
        id=node_id,
        kind=NodeKind.File,
        name=name,
        file_path=path,
        start_line=1,
        end_line=100,
        language="kotlin",
    )


def _make_screen_node(node_id: str, name: str, path: str) -> GraphNode:
    return GraphNode(
        id=node_id,
        kind=NodeKind.Screen,
        name=name,
        file_path=path,
        start_line=1,
        end_line=80,
        language="kotlin",
    )


@pytest.fixture()
def store(tmp_path: Path) -> LadybugGraphStore:
    s = LadybugGraphStore(tmp_path / "graph.db")
    s.create_schema()
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_create_schema_is_idempotent(tmp_path: Path) -> None:
    """Calling create_schema twice must not raise."""
    with LadybugGraphStore(tmp_path / "graph.db") as s:
        s.create_schema()
        s.create_schema()  # second call must succeed


# ---------------------------------------------------------------------------
# Node round-trip
# ---------------------------------------------------------------------------


def test_add_and_retrieve_nodes(store: LadybugGraphStore) -> None:
    nodes = [
        _make_file_node("f1", "MainActivity.kt", "/src/MainActivity.kt"),
        _make_file_node("f2", "AuthActivity.kt", "/src/AuthActivity.kt"),
    ]
    store.add_nodes(nodes)

    rows = store.query("MATCH (n:KlitNode) RETURN n.id, n.name ORDER BY n.id")
    ids = [r[0] for r in rows]
    names = [r[1] for r in rows]
    assert "f1" in ids
    assert "f2" in ids
    assert "MainActivity.kt" in names
    assert "AuthActivity.kt" in names


def test_node_kind_is_stored(store: LadybugGraphStore) -> None:
    store.add_nodes([_make_screen_node("s1", "HomeScreen", "/src/HomeScreen.kt")])
    rows = store.query("MATCH (n:KlitNode {id: 's1'}) RETURN n.kind")
    assert rows == [["Screen"]]


def test_node_metadata_is_stored(store: LadybugGraphStore) -> None:
    store.add_nodes([_make_file_node("f1", "Main.kt", "/src/Main.kt")])
    rows = store.query(
        "MATCH (n:KlitNode {id: 'f1'}) RETURN n.file_path, n.start_line, n.end_line, n.language"
    )
    assert rows == [["/src/Main.kt", 1, 100, "kotlin"]]


def test_add_nodes_upserts_on_duplicate_id(store: LadybugGraphStore) -> None:
    """Second add_nodes call with the same ID must overwrite, not error."""
    n = _make_file_node("f1", "Old.kt", "/src/Old.kt")
    store.add_nodes([n])
    updated = GraphNode(
        id="f1",
        kind=NodeKind.File,
        name="New.kt",
        file_path="/src/New.kt",
        start_line=1,
        end_line=50,
        language="kotlin",
    )
    store.add_nodes([updated])
    rows = store.query("MATCH (n:KlitNode) WHERE n.id = 'f1' RETURN n.name")
    assert rows == [["New.kt"]]


def test_filter_by_node_kind(store: LadybugGraphStore) -> None:
    store.add_nodes(
        [
            _make_file_node("f1", "A.kt", "/src/A.kt"),
            _make_screen_node("s1", "Home", "/src/Home.kt"),
        ]
    )
    rows = store.query("MATCH (n:KlitNode) WHERE n.kind = 'Screen' RETURN n.id")
    assert rows == [["s1"]]


# ---------------------------------------------------------------------------
# Edge round-trip
# ---------------------------------------------------------------------------


def test_add_and_retrieve_imports_edge(store: LadybugGraphStore) -> None:
    store.add_nodes(
        [
            _make_file_node("f1", "Main.kt", "/src/Main.kt"),
            _make_file_node("f2", "Auth.kt", "/src/Auth.kt"),
        ]
    )
    store.add_edges(
        [
            GraphEdge(
                src_id="f1",
                dst_id="f2",
                type=RelationType.IMPORTS,
                confidence=1.0,
                file_path="/src/Main.kt",
                line=3,
            )
        ]
    )
    rows = store.query(
        "MATCH (a:KlitNode)-[e:KlitEdge]->(b:KlitNode) "
        "WHERE e.type = 'IMPORTS' RETURN a.id, b.id, e.confidence"
    )
    assert rows == [["f1", "f2", 1.0]]


def test_query_what_imports_x(store: LadybugGraphStore) -> None:
    """Core acceptance criterion: Cypher query 'what imports X'."""
    store.add_nodes(
        [
            _make_file_node("f1", "Main.kt", "/src/Main.kt"),
            _make_file_node("f2", "Auth.kt", "/src/Auth.kt"),
            _make_file_node("f3", "Utils.kt", "/src/Utils.kt"),
        ]
    )
    store.add_edges(
        [
            GraphEdge(src_id="f1", dst_id="f2", type=RelationType.IMPORTS, confidence=1.0),
            GraphEdge(src_id="f3", dst_id="f2", type=RelationType.IMPORTS, confidence=1.0),
        ]
    )
    # "What imports Auth.kt?"
    rows = store.query(
        "MATCH (a:KlitNode)-[e:KlitEdge]->(b:KlitNode {id: 'f2'}) "
        "WHERE e.type = 'IMPORTS' RETURN a.name ORDER BY a.name"
    )
    names = [r[0] for r in rows]
    assert names == ["Main.kt", "Utils.kt"]


def test_query_navigates_to_edges(store: LadybugGraphStore) -> None:
    store.add_nodes(
        [
            _make_screen_node("s1", "HomeActivity", "/src/HomeActivity.kt"),
            _make_screen_node("s2", "AuthActivity", "/src/AuthActivity.kt"),
        ]
    )
    store.add_edges(
        [
            GraphEdge(
                src_id="s1",
                dst_id="s2",
                type=RelationType.NAVIGATES_TO,
                confidence=0.95,
                trigger="button_tap",
            )
        ]
    )
    rows = store.query(
        "MATCH (a:KlitNode)-[e:KlitEdge]->(b:KlitNode) "
        "WHERE e.type = 'NAVIGATES_TO' RETURN a.name, b.name, e.trigger, e.confidence"
    )
    assert rows == [["HomeActivity", "AuthActivity", "button_tap", 0.95]]


def test_add_edge_with_missing_src_is_silent(store: LadybugGraphStore) -> None:
    """Edge whose src doesn't exist must not raise — just produce no row."""
    store.add_nodes([_make_file_node("f2", "Auth.kt", "/src/Auth.kt")])
    store.add_edges(
        [GraphEdge(src_id="missing", dst_id="f2", type=RelationType.IMPORTS, confidence=1.0)]
    )
    rows = store.query("MATCH (a:KlitNode)-[:KlitEdge]->(b:KlitNode) RETURN a.id, b.id")
    assert rows == []


def test_add_edge_with_missing_dst_is_silent(store: LadybugGraphStore) -> None:
    store.add_nodes([_make_file_node("f1", "Main.kt", "/src/Main.kt")])
    store.add_edges(
        [GraphEdge(src_id="f1", dst_id="missing", type=RelationType.IMPORTS, confidence=1.0)]
    )
    rows = store.query("MATCH (a:KlitNode)-[:KlitEdge]->(b:KlitNode) RETURN a.id, b.id")
    assert rows == []


def test_add_edges_survives_hostile_text(store: LadybugGraphStore) -> None:
    """Edge text with commas, quotes, backslashes, newlines and NUL must load.

    The bulk COPY path round-trips arbitrary source-derived text through CSV;
    this guards against the "expected N values per row, but got more" / NUL
    COPY failures by quoting every field and stripping control characters.
    """
    from klit_flow.graph.model import ConditionLevel

    store.add_nodes(
        [
            _make_file_node("f1", "Main.kt", "/src/Main.kt"),
            _make_file_node("f2", "Auth.kt", "/src/Auth.kt"),
        ]
    )
    store.add_edges(
        [
            GraphEdge(
                src_id="f1",
                dst_id="f2",
                type=RelationType.CALLS,
                confidence=0.5,
                file_path="a\x00b.kt",  # NUL would otherwise abort the COPY
                line=7,
                trigger="tap\nnow",  # embedded newline
                conditions=[
                    # commas inside generics + escaped quotes + backslash
                    ConditionLevel(expression='x is Map<String, Int> && s == "a,b\\"c"', kind="if")
                ],
            )
        ]
    )
    rows = store.query(
        "MATCH (a:KlitNode)-[e:KlitEdge]->(b:KlitNode) RETURN a.id, b.id, e.condition"
    )
    assert len(rows) == 1
    assert rows[0][0] == "f1" and rows[0][1] == "f2"
    # The generics comma survived intact (no row-splitting).
    assert "Map<String, Int>" in rows[0][2]


def test_multiple_edge_types_stored(store: LadybugGraphStore) -> None:
    store.add_nodes(
        [
            _make_file_node("f1", "A.kt", "/A.kt"),
            _make_file_node("f2", "B.kt", "/B.kt"),
        ]
    )
    store.add_edges(
        [
            GraphEdge(src_id="f1", dst_id="f2", type=RelationType.IMPORTS, confidence=1.0),
            GraphEdge(src_id="f1", dst_id="f2", type=RelationType.CALLS, confidence=0.7),
        ]
    )
    rows = store.query(
        "MATCH (a:KlitNode)-[e:KlitEdge]->(b:KlitNode) RETURN e.type ORDER BY e.type"
    )
    types = [r[0] for r in rows]
    assert "CALLS" in types
    assert "IMPORTS" in types


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager_closes_store(tmp_path: Path) -> None:
    with LadybugGraphStore(tmp_path / "graph.db") as s:
        s.create_schema()
        s.add_nodes([_make_file_node("f1", "A.kt", "/A.kt")])
    assert s._closed  # noqa: SLF001


# ---------------------------------------------------------------------------
# ABC contract: callers must be able to use GraphStore type, not concrete class
# ---------------------------------------------------------------------------


def test_graphstore_is_abstract() -> None:
    """GraphStore ABC cannot be instantiated directly."""
    with pytest.raises(TypeError):
        GraphStore()  # type: ignore[abstract]


def test_ladybug_store_is_graphstore_instance(tmp_path: Path) -> None:
    with LadybugGraphStore(tmp_path / "graph.db") as s:
        assert isinstance(s, GraphStore)


# ---------------------------------------------------------------------------
# Vector round-trip
# ---------------------------------------------------------------------------


def test_upsert_vectors_stores_embedding(store4: LadybugGraphStore) -> None:
    """upsert_vectors must not raise and the node must be retrievable after."""
    store4.add_nodes([_make_file_node("f1", "Main.kt", "/src/Main.kt")])
    store4.upsert_vectors(["f1"], [[0.1, 0.2, 0.3, 0.4]])
    rows = store4.query("MATCH (n:KlitNode {id: 'f1'}) RETURN n.id")
    assert rows == [["f1"]]


@pytest.fixture()
def store4(tmp_path: Path) -> LadybugGraphStore:
    s = LadybugGraphStore(tmp_path / "graph4.db", emb_dim=4)
    s.create_schema()
    yield s
    s.close()


def test_vector_search_returns_nearest(store4: LadybugGraphStore) -> None:
    """Nearest-neighbour search should rank the most similar node first."""
    nodes = [
        _make_file_node("f1", "Main.kt", "/src/Main.kt"),
        _make_file_node("f2", "Auth.kt", "/src/Auth.kt"),
    ]
    store4.add_nodes(nodes)
    store4.upsert_vectors(
        ["f1", "f2"],
        [
            [1.0, 0.0, 0.0, 0.0],  # f1: points along x
            [0.0, 1.0, 0.0, 0.0],  # f2: points along y
        ],
    )
    # Query vector near f1
    results = store4.vector_search([1.0, 0.0, 0.0, 0.0], k=2)
    assert results[0] == "f1"


def test_vector_search_top_k(store4: LadybugGraphStore) -> None:
    nodes = [_make_file_node(f"f{i}", f"F{i}.kt", f"/F{i}.kt") for i in range(5)]
    store4.add_nodes(nodes)
    embeddings = [[float(i == j) for j in range(4)] for i in range(5)]
    store4.upsert_vectors([f"f{i}" for i in range(5)], embeddings[:4])  # only first 4 have vecs
    results = store4.vector_search([1.0, 0.0, 0.0, 0.0], k=2)
    assert len(results) <= 2
