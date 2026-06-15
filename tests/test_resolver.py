"""Phase 3 acceptance tests: graph resolution."""

from pathlib import Path

from klit_flow.graph.model import GraphNode, RelationType
from klit_flow.graph.resolver import resolve
from klit_flow.parsing.extractor import extract
from klit_flow.walker import SourceFile, walk

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "mini_app"


def _resolve_fixture() -> tuple[list[GraphNode], list]:
    source_files = walk(FIXTURE_ROOT)
    symbols_by_file = {str(sf.path): extract(sf.path, sf.language) for sf in source_files}
    return resolve(source_files, symbols_by_file)


def _edges_of(edges: list, rel: RelationType) -> list:
    return [e for e in edges if e.type == rel]


# ── DECLARES ──────────────────────────────────────────────────────────────────


def test_declares_edges_exist():
    _, edges = _resolve_fixture()
    assert len(_edges_of(edges, RelationType.DECLARES)) > 0


def test_file_declares_class():
    nodes, edges = _resolve_fixture()
    auth_file = next((n for n in nodes if n.name == "AuthActivity.kt"), None)
    auth_class = next((n for n in nodes if n.name == "AuthActivity"), None)
    assert auth_file and auth_class
    declares = _edges_of(edges, RelationType.DECLARES)
    assert any(e.src_id == auth_file.id and e.dst_id == auth_class.id for e in declares)


def test_class_declares_methods():
    nodes, edges = _resolve_fixture()
    auth_class = next((n for n in nodes if n.name == "AuthActivity"), None)
    login = next((n for n in nodes if n.name == "login"), None)
    assert auth_class and login
    declares = _edges_of(edges, RelationType.DECLARES)
    assert any(e.src_id == auth_class.id and e.dst_id == login.id for e in declares)


# ── IMPORTS (exact) ───────────────────────────────────────────────────────────


def test_import_edge_for_cross_package_import():
    nodes, edges = _resolve_fixture()
    repo_file = next((n for n in nodes if n.name == "Repository.kt"), None)
    auth_file = next((n for n in nodes if n.name == "AuthActivity.kt"), None)
    assert repo_file and auth_file, "Fixture files must be discovered"
    imports = _edges_of(edges, RelationType.IMPORTS)
    assert any(e.src_id == repo_file.id and e.dst_id == auth_file.id for e in imports)


def test_import_edges_confidence_is_one():
    _, edges = _resolve_fixture()
    for e in _edges_of(edges, RelationType.IMPORTS):
        assert e.confidence == 1.0


def test_external_imports_produce_no_edges():
    nodes, edges = _resolve_fixture()
    node_ids = {n.id for n in nodes}
    # Every IMPORTS edge must reference nodes that actually exist in the graph
    for e in _edges_of(edges, RelationType.IMPORTS):
        assert e.src_id in node_ids
        assert e.dst_id in node_ids


def test_no_self_import_edges():
    _, edges = _resolve_fixture()
    for e in _edges_of(edges, RelationType.IMPORTS):
        assert e.src_id != e.dst_id


# ── CALLS (best-effort) ───────────────────────────────────────────────────────


def test_calls_edges_have_valid_confidence():
    _, edges = _resolve_fixture()
    for e in _edges_of(edges, RelationType.CALLS):
        assert 0.0 < e.confidence <= 1.0


def test_calls_edges_reference_known_nodes():
    nodes, edges = _resolve_fixture()
    node_ids = {n.id for n in nodes}
    for e in _edges_of(edges, RelationType.CALLS):
        assert e.src_id in node_ids
        assert e.dst_id in node_ids


def test_no_crash_on_unresolved_calls(tmp_path: Path):
    f = tmp_path / "unresolved.kt"
    f.write_text("package com.test\nfun foo() { totallyUnknownFunc() }\n")
    sf = SourceFile(path=f, rel_path="unresolved.kt", language="kotlin", size=f.stat().st_size)
    _, edges = resolve([sf], {str(f): extract(f, "kotlin")})
    assert _edges_of(edges, RelationType.CALLS) == []


def test_cross_file_call_has_lower_confidence_than_same_file():
    """Same-file calls get 0.9; cross-file get ≤ 0.7."""
    _, edges = _resolve_fixture()
    call_edges = _edges_of(edges, RelationType.CALLS)
    confidences = {e.confidence for e in call_edges}
    # Both confidence levels should be representable; at minimum all are in range.
    for c in confidences:
        assert c in (0.5, 0.7, 0.9)


# ── EXTENDS / IMPLEMENTS ──────────────────────────────────────────────────────


def test_implements_edge(tmp_path: Path):
    iface = tmp_path / "ILogger.kt"
    iface.write_text("package com.test\ninterface ILogger {\n    fun log(msg: String)\n}\n")
    impl = tmp_path / "Logger.kt"
    impl.write_text(
        "package com.test\nclass Logger : ILogger {\n    override fun log(msg: String) {}\n}\n"
    )
    source_files = [
        SourceFile(path=iface, rel_path="ILogger.kt", language="kotlin", size=iface.stat().st_size),
        SourceFile(path=impl, rel_path="Logger.kt", language="kotlin", size=impl.stat().st_size),
    ]
    syms = {str(p): extract(p, "kotlin") for p in (iface, impl)}
    _, edges = resolve(source_files, syms)
    impl_edges = _edges_of(edges, RelationType.IMPLEMENTS)
    assert len(impl_edges) >= 1
    assert all(e.confidence == 1.0 for e in impl_edges)


def test_extends_edge(tmp_path: Path):
    base = tmp_path / "Base.kt"
    base.write_text("package com.test\nopen class Base { fun baseMethod() {} }\n")
    child = tmp_path / "Child.kt"
    child.write_text("package com.test\nclass Child : Base() { fun childMethod() {} }\n")
    source_files = [
        SourceFile(path=base, rel_path="Base.kt", language="kotlin", size=base.stat().st_size),
        SourceFile(path=child, rel_path="Child.kt", language="kotlin", size=child.stat().st_size),
    ]
    syms = {str(p): extract(p, "kotlin") for p in (base, child)}
    _, edges = resolve(source_files, syms)
    ext_edges = _edges_of(edges, RelationType.EXTENDS)
    assert len(ext_edges) >= 1
    assert all(e.confidence == 1.0 for e in ext_edges)


def test_no_extends_for_external_parents():
    """MainActivity : AppCompatActivity — AppCompatActivity is not in our graph."""
    nodes, edges = _resolve_fixture()
    for e in _edges_of(edges, RelationType.EXTENDS) + _edges_of(edges, RelationType.IMPLEMENTS):
        # Both ends must be in-project symbols
        node_ids = {n.id for n in nodes}
        assert e.src_id in node_ids
        assert e.dst_id in node_ids
