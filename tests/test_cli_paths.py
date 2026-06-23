"""Tests for relativizing stored file paths so the index is portable."""

from __future__ import annotations

from pathlib import Path

from klit_flow.cli import _relativize_edges, _relativize_nodes, _relativize_path
from klit_flow.graph.model import GraphEdge, GraphNode, NodeKind, RelationType

_ROOT = Path("/home/me/project").resolve()


def _abs(*parts: str) -> str:
    return str(_ROOT.joinpath(*parts))


def test_relativize_absolute_under_root() -> None:
    assert _relativize_path(_abs("app", "src", "Main.kt"), _ROOT) == "app/src/Main.kt"


def test_relativize_already_relative_is_normalized() -> None:
    assert _relativize_path("app/src/Main.kt", _ROOT) == "app/src/Main.kt"


def test_relativize_empty_and_none() -> None:
    assert _relativize_path("", _ROOT) == ""
    assert _relativize_path(None, _ROOT) is None


def test_relativize_outside_root_left_absolute() -> None:
    other = Path("/somewhere/else/File.kt").resolve()
    out = _relativize_path(str(other), _ROOT)
    # Not under root → not made relative (stays absolute, POSIX-normalized).
    assert not out.startswith("app")
    assert "File.kt" in out


def test_relativize_nodes_copies_and_rewrites() -> None:
    node = GraphNode(
        id="n1",
        kind=NodeKind.File,
        name="Main.kt",
        file_path=_abs("app", "Main.kt"),
        start_line=1,
        end_line=10,
        language="kotlin",
    )
    out = _relativize_nodes([node], _ROOT)
    assert out[0].file_path == "app/Main.kt"
    assert node.file_path != out[0].file_path  # original untouched (copy)


def test_relativize_edges_handles_none_file_path() -> None:
    edge = GraphEdge(src_id="a", dst_id="b", type=RelationType.CALLS, file_path=None)
    out = _relativize_edges([edge], _ROOT)
    assert out[0].file_path is None

    edge2 = GraphEdge(
        src_id="a",
        dst_id="b",
        type=RelationType.NAVIGATES_TO,
        file_path=_abs("app", "Nav.kt"),
    )
    out2 = _relativize_edges([edge2], _ROOT)
    assert out2[0].file_path == "app/Nav.kt"
