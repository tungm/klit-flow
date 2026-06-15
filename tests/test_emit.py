"""Phase 4 acceptance tests: JSON, Markdown, and Mermaid emitters."""

import json
from pathlib import Path

import frontmatter

from klit_flow.emit.json_emitter import emit_graph_json
from klit_flow.emit.markdown_emitter import emit_module_docs
from klit_flow.emit.mermaid_emitter import emit_dependency_diagram
from klit_flow.graph.model import GraphNode
from klit_flow.graph.resolver import resolve
from klit_flow.parsing.extractor import extract
from klit_flow.walker import walk

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "mini_app"


def _build_graph() -> tuple[list[GraphNode], list]:
    source_files = walk(FIXTURE_ROOT)
    symbols_by_file = {str(sf.path): extract(sf.path, sf.language) for sf in source_files}
    return resolve(source_files, symbols_by_file)


# ── graph.json ────────────────────────────────────────────────────────────────


def test_graph_json_written(tmp_path: Path):
    nodes, edges = _build_graph()
    out = emit_graph_json(nodes, edges, tmp_path)
    assert out.exists() and out.name == "graph.json"


def test_graph_json_top_level_keys(tmp_path: Path):
    nodes, edges = _build_graph()
    emit_graph_json(nodes, edges, tmp_path)
    data = json.loads((tmp_path / "graph.json").read_text())
    assert set(data.keys()) == {"nodes", "edges"}


def test_graph_json_node_fields(tmp_path: Path):
    nodes, edges = _build_graph()
    emit_graph_json(nodes, edges, tmp_path)
    data = json.loads((tmp_path / "graph.json").read_text())
    required = {"id", "kind", "name", "file_path", "start_line", "end_line", "language"}
    for n in data["nodes"]:
        assert required <= n.keys()


def test_graph_json_edge_fields(tmp_path: Path):
    nodes, edges = _build_graph()
    emit_graph_json(nodes, edges, tmp_path)
    data = json.loads((tmp_path / "graph.json").read_text())
    required = {"src_id", "dst_id", "type", "confidence"}
    for e in data["edges"]:
        assert required <= e.keys()


def test_graph_json_counts_match(tmp_path: Path):
    nodes, edges = _build_graph()
    emit_graph_json(nodes, edges, tmp_path)
    data = json.loads((tmp_path / "graph.json").read_text())
    assert len(data["nodes"]) == len(nodes)
    assert len(data["edges"]) == len(edges)


def test_graph_json_valid_node_kinds(tmp_path: Path):
    nodes, edges = _build_graph()
    emit_graph_json(nodes, edges, tmp_path)
    data = json.loads((tmp_path / "graph.json").read_text())
    valid = {"File", "Module", "Function", "Class", "Method", "Interface", "Screen"}
    for n in data["nodes"]:
        assert n["kind"] in valid


def test_graph_json_confidence_in_range(tmp_path: Path):
    nodes, edges = _build_graph()
    emit_graph_json(nodes, edges, tmp_path)
    data = json.loads((tmp_path / "graph.json").read_text())
    for e in data["edges"]:
        assert 0.0 <= e["confidence"] <= 1.0


def test_graph_json_idempotent(tmp_path: Path):
    nodes, edges = _build_graph()
    emit_graph_json(nodes, edges, tmp_path)
    first = (tmp_path / "graph.json").read_text()
    emit_graph_json(nodes, edges, tmp_path)
    second = (tmp_path / "graph.json").read_text()
    assert first == second


# ── Markdown module docs ──────────────────────────────────────────────────────


def test_module_docs_written(tmp_path: Path):
    nodes, edges = _build_graph()
    paths = emit_module_docs(nodes, edges, tmp_path)
    assert len(paths) > 0
    for p in paths:
        assert p.exists() and p.suffix == ".md"


def test_module_docs_in_correct_directory(tmp_path: Path):
    nodes, edges = _build_graph()
    paths = emit_module_docs(nodes, edges, tmp_path)
    expected_dir = tmp_path / "docs" / "modules"
    for p in paths:
        assert p.parent == expected_dir


def test_module_doc_frontmatter_required_fields(tmp_path: Path):
    nodes, edges = _build_graph()
    paths = emit_module_docs(nodes, edges, tmp_path)
    required = {"id", "kind", "path", "depends_on", "symbols"}
    for p in paths:
        post = frontmatter.load(str(p))
        assert required <= post.metadata.keys(), f"Missing fields in {p.name}"


def test_module_doc_kind_is_module(tmp_path: Path):
    nodes, edges = _build_graph()
    paths = emit_module_docs(nodes, edges, tmp_path)
    for p in paths:
        assert frontmatter.load(str(p))["kind"] == "Module"


def test_module_doc_depends_on_and_symbols_are_lists(tmp_path: Path):
    nodes, edges = _build_graph()
    paths = emit_module_docs(nodes, edges, tmp_path)
    for p in paths:
        post = frontmatter.load(str(p))
        assert isinstance(post["depends_on"], list)
        assert isinstance(post["symbols"], list)


def test_module_doc_one_file_per_source_file(tmp_path: Path):
    nodes, edges = _build_graph()
    paths = emit_module_docs(nodes, edges, tmp_path)
    from klit_flow.graph.model import NodeKind

    file_node_count = sum(1 for n in nodes if n.kind == NodeKind.File)
    assert len(paths) == file_node_count


def test_module_doc_repository_depends_on_authactivity(tmp_path: Path):
    nodes, edges = _build_graph()
    paths = emit_module_docs(nodes, edges, tmp_path)
    auth_node = next((n for n in nodes if n.name == "AuthActivity.kt"), None)
    repo_doc = next((p for p in paths if "Repository" in p.name), None)
    assert auth_node and repo_doc
    post = frontmatter.load(str(repo_doc))
    assert auth_node.id in post["depends_on"]


def test_module_doc_body_is_nonempty(tmp_path: Path):
    nodes, edges = _build_graph()
    paths = emit_module_docs(nodes, edges, tmp_path)
    for p in paths:
        assert frontmatter.load(str(p)).content.strip()


def test_module_doc_ids_match_graph_nodes(tmp_path: Path):
    nodes, edges = _build_graph()
    paths = emit_module_docs(nodes, edges, tmp_path)
    node_ids = {n.id for n in nodes}
    for p in paths:
        post = frontmatter.load(str(p))
        assert post["id"] in node_ids


# ── Mermaid dependency diagram ────────────────────────────────────────────────


def test_mermaid_written(tmp_path: Path):
    nodes, edges = _build_graph()
    out = emit_dependency_diagram(nodes, edges, tmp_path)
    assert out.exists() and out.name == "dependencies.mmd"


def test_mermaid_in_diagrams_directory(tmp_path: Path):
    nodes, edges = _build_graph()
    out = emit_dependency_diagram(nodes, edges, tmp_path)
    assert out.parent == tmp_path / "diagrams"


def test_mermaid_starts_with_flowchart(tmp_path: Path):
    nodes, edges = _build_graph()
    out = emit_dependency_diagram(nodes, edges, tmp_path)
    assert out.read_text().startswith("flowchart")


def test_mermaid_contains_arrow(tmp_path: Path):
    nodes, edges = _build_graph()
    out = emit_dependency_diagram(nodes, edges, tmp_path)
    assert "-->" in out.read_text()


def test_mermaid_references_real_file_names(tmp_path: Path):
    nodes, edges = _build_graph()
    out = emit_dependency_diagram(nodes, edges, tmp_path)
    content = out.read_text()
    # Repository.kt → AuthActivity.kt is the known import in the fixture
    assert "Repository.kt" in content
    assert "AuthActivity.kt" in content


def test_mermaid_no_duplicate_node_declarations(tmp_path: Path):
    nodes, edges = _build_graph()
    out = emit_dependency_diagram(nodes, edges, tmp_path)
    declarations = [ln for ln in out.read_text().splitlines() if "[" in ln and "]" in ln]
    node_ids = [ln.strip().split("[")[0] for ln in declarations]
    assert len(node_ids) == len(set(node_ids))
