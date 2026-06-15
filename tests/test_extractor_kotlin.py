"""Phase 2 acceptance tests: Kotlin symbol extraction + provisional graph.json."""

import hashlib
import json
from pathlib import Path

from klit_flow.graph.model import GraphEdge, GraphNode, NodeKind, RelationType
from klit_flow.parsing.extractor import Symbol, extract

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "mini_app"
MAIN = FIXTURE_ROOT / "src" / "MainActivity.kt"
AUTH = FIXTURE_ROOT / "src" / "AuthActivity.kt"
UTILS = FIXTURE_ROOT / "src" / "Utils.kt"


def _by_kind(symbols: list[Symbol], kind: str) -> list[Symbol]:
    return [s for s in symbols if s.kind == kind]


def _names(symbols: list[Symbol], kind: str) -> set[str]:
    return {s.name for s in _by_kind(symbols, kind)}


# ── MainActivity.kt ────────────────────────────────────────────────────────────


def test_main_class_detected():
    assert "MainActivity" in _names(extract(MAIN, "kotlin"), "Class")


def test_main_on_create_is_method():
    assert "onCreate" in _names(extract(MAIN, "kotlin"), "Method")


def test_main_imports():
    imports = _names(extract(MAIN, "kotlin"), "Import")
    assert "android.os.Bundle" in imports
    assert "androidx.appcompat.app.AppCompatActivity" in imports


# ── AuthActivity.kt ────────────────────────────────────────────────────────────


def test_auth_class_detected():
    assert "AuthActivity" in _names(extract(AUTH, "kotlin"), "Class")


def test_auth_methods():
    methods = _names(extract(AUTH, "kotlin"), "Method")
    assert "onCreate" in methods
    assert "login" in methods


def test_auth_no_top_level_functions():
    assert _names(extract(AUTH, "kotlin"), "Function") == set()


# ── Utils.kt ──────────────────────────────────────────────────────────────────


def test_utils_top_level_functions():
    funcs = _names(extract(UTILS, "kotlin"), "Function")
    assert "formatDate" in funcs
    assert "isValidEmail" in funcs


def test_utils_no_classes():
    assert _names(extract(UTILS, "kotlin"), "Class") == set()


def test_utils_no_methods():
    assert _names(extract(UTILS, "kotlin"), "Method") == set()


# ── Line numbers ───────────────────────────────────────────────────────────────


def test_line_numbers_are_positive():
    for sym in extract(AUTH, "kotlin"):
        assert sym.start_line >= 1
        assert sym.end_line >= sym.start_line


# ── ID stability and uniqueness ───────────────────────────────────────────────


def test_ids_are_stable():
    ids1 = {s.id for s in extract(UTILS, "kotlin")}
    ids2 = {s.id for s in extract(UTILS, "kotlin")}
    assert ids1 == ids2


def test_ids_are_unique():
    symbols = extract(AUTH, "kotlin")
    ids = [s.id for s in symbols]
    assert len(ids) == len(set(ids))


# ── Error resilience ──────────────────────────────────────────────────────────


def test_missing_file_returns_empty(tmp_path: Path):
    assert extract(tmp_path / "nope.kt", "kotlin") == []


def test_unsupported_language_returns_empty(tmp_path: Path):
    f = tmp_path / "hello.rb"
    f.write_text("puts 'hello'")
    assert extract(f, "ruby") == []


def test_malformed_source_returns_empty(tmp_path: Path):
    # tree-sitter is resilient but the extractor must never raise
    f = tmp_path / "broken.kt"
    f.write_bytes(b"\xff\xfe broken \x00 garbage")
    result = extract(f, "kotlin")
    assert isinstance(result, list)


# ── Provisional graph.json ────────────────────────────────────────────────────


def _file_node(source_file: Path, language: str) -> GraphNode:
    fid = hashlib.sha256(f"File:{source_file}".encode()).hexdigest()[:16]
    return GraphNode(
        id=fid,
        kind=NodeKind.File,
        name=source_file.name,
        file_path=str(source_file),
        start_line=1,
        end_line=0,
        language=language,
    )


def test_provisional_graph_json(tmp_path: Path):
    symbols = extract(UTILS, "kotlin")
    file_node = _file_node(UTILS, "kotlin")

    sym_nodes = [
        GraphNode(
            id=s.id,
            kind=NodeKind(s.kind),
            name=s.name,
            file_path=s.file_path,
            start_line=s.start_line,
            end_line=s.end_line,
            language=s.language,
        )
        for s in symbols
        if s.kind != "Import"
    ]

    edges = [
        GraphEdge(
            src_id=file_node.id,
            dst_id=n.id,
            type=RelationType.DECLARES,
            file_path=str(UTILS),
        )
        for n in sym_nodes
    ]

    graph = {
        "nodes": [n.model_dump() for n in [file_node, *sym_nodes]],
        "edges": [e.model_dump() for e in edges],
    }

    out = tmp_path / "graph.json"
    out.write_text(json.dumps(graph, indent=2))
    loaded = json.loads(out.read_text())

    assert len(loaded["nodes"]) >= 3  # File + formatDate + isValidEmail
    assert all("id" in n and "kind" in n and "name" in n for n in loaded["nodes"])
    assert all(e["type"] == "DECLARES" for e in loaded["edges"])
    assert loaded["nodes"][0]["kind"] == "File"
