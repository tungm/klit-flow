"""Tests for `klit-flow flows` CLI command."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from klit_flow.cli import app
from klit_flow.graph.model import GraphEdge, GraphNode, NodeKind, RelationType
from klit_flow.graph.store import LadybugGraphStore

runner = CliRunner()

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MAIN = GraphNode(
    id="main",
    kind=NodeKind.Screen,
    name="MainActivity",
    file_path="/app/src/MainActivity.kt",
    start_line=1,
    end_line=10,
    language="kotlin",
)
_AUTH = GraphNode(
    id="auth",
    kind=NodeKind.Screen,
    name="AuthActivity",
    file_path="/app/src/AuthActivity.kt",
    start_line=1,
    end_line=10,
    language="kotlin",
)
_PROFILE = GraphNode(
    id="profile",
    kind=NodeKind.Screen,
    name="ProfileActivity",
    file_path="/app/src/ProfileActivity.kt",
    start_line=1,
    end_line=10,
    language="kotlin",
)

_EDGE_MAIN_AUTH = GraphEdge(
    src_id="main",
    dst_id="auth",
    type=RelationType.NAVIGATES_TO,
    confidence=0.95,
    file_path="/app/src/MainActivity.kt",
    line=5,
    trigger="button_tap",
)
_EDGE_AUTH_PROFILE = GraphEdge(
    src_id="auth",
    dst_id="profile",
    type=RelationType.NAVIGATES_TO,
    confidence=0.95,
    file_path="/app/src/AuthActivity.kt",
    line=8,
    trigger="programmatic",
)


@pytest.fixture()
def indexed_dir(tmp_path: Path) -> Path:
    """Return a temp dir with a minimal .klit-flow/graph.db already built."""
    klit_dir = tmp_path / ".klit-flow"
    klit_dir.mkdir()
    db_path = klit_dir / "graph.db"
    with LadybugGraphStore(db_path) as store:
        store.create_schema()
        store.add_nodes([_MAIN, _AUTH, _PROFILE])
        store.add_edges([_EDGE_MAIN_AUTH, _EDGE_AUTH_PROFILE])
    return tmp_path


def _activity_lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if "Activity" in line]


# ---------------------------------------------------------------------------
# All flows (no filter)
# ---------------------------------------------------------------------------


def test_flows_all_edges(monkeypatch: pytest.MonkeyPatch, indexed_dir: Path) -> None:
    monkeypatch.chdir(indexed_dir)
    result = runner.invoke(app, ["flows"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "MainActivity" in result.output
    assert "AuthActivity" in result.output
    assert "ProfileActivity" in result.output


def test_flows_all_shows_trigger(monkeypatch: pytest.MonkeyPatch, indexed_dir: Path) -> None:
    monkeypatch.chdir(indexed_dir)
    result = runner.invoke(app, ["flows"], catch_exceptions=False)
    assert "button_tap" in result.output
    assert "programmatic" in result.output


def test_flows_all_shows_confidence(monkeypatch: pytest.MonkeyPatch, indexed_dir: Path) -> None:
    monkeypatch.chdir(indexed_dir)
    result = runner.invoke(app, ["flows"], catch_exceptions=False)
    assert "0.95" in result.output


def test_flows_all_shows_header_row(monkeypatch: pytest.MonkeyPatch, indexed_dir: Path) -> None:
    monkeypatch.chdir(indexed_dir)
    result = runner.invoke(app, ["flows"], catch_exceptions=False)
    assert "FROM" in result.output
    assert "TO" in result.output
    assert "TRIGGER" in result.output


def test_flows_all_two_edges(monkeypatch: pytest.MonkeyPatch, indexed_dir: Path) -> None:
    monkeypatch.chdir(indexed_dir)
    result = runner.invoke(app, ["flows"], catch_exceptions=False)
    assert len(_activity_lines(result.output)) == 2


# ---------------------------------------------------------------------------
# Filtered flows
# ---------------------------------------------------------------------------


def test_flows_filter_auth_shows_both_edges(
    monkeypatch: pytest.MonkeyPatch, indexed_dir: Path
) -> None:
    monkeypatch.chdir(indexed_dir)
    result = runner.invoke(app, ["flows", "AuthActivity"], catch_exceptions=False)
    assert result.exit_code == 0
    # main→auth (AuthActivity is dst) and auth→profile (AuthActivity is src)
    assert len(_activity_lines(result.output)) == 2


def test_flows_filter_main_shows_one_edge(
    monkeypatch: pytest.MonkeyPatch, indexed_dir: Path
) -> None:
    monkeypatch.chdir(indexed_dir)
    result = runner.invoke(app, ["flows", "MainActivity"], catch_exceptions=False)
    assert result.exit_code == 0
    lines = _activity_lines(result.output)
    assert len(lines) == 1
    assert "MainActivity" in lines[0]
    assert "AuthActivity" in lines[0]


def test_flows_filter_profile_shows_one_edge(
    monkeypatch: pytest.MonkeyPatch, indexed_dir: Path
) -> None:
    monkeypatch.chdir(indexed_dir)
    result = runner.invoke(app, ["flows", "ProfileActivity"], catch_exceptions=False)
    assert result.exit_code == 0
    lines = _activity_lines(result.output)
    assert len(lines) == 1
    assert "ProfileActivity" in lines[0]


def test_flows_filter_unknown_screen(monkeypatch: pytest.MonkeyPatch, indexed_dir: Path) -> None:
    monkeypatch.chdir(indexed_dir)
    result = runner.invoke(app, ["flows", "UnknownScreen"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "No navigation edges found" in result.output


# ---------------------------------------------------------------------------
# Empty graph
# ---------------------------------------------------------------------------


def test_flows_empty_graph(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    klit_dir = tmp_path / ".klit-flow"
    klit_dir.mkdir()
    db_path = klit_dir / "graph.db"
    with LadybugGraphStore(db_path) as store:
        store.create_schema()
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["flows"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "No navigation edges found" in result.output


# ---------------------------------------------------------------------------
# Missing index
# ---------------------------------------------------------------------------


def test_flows_missing_index_exits_nonzero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["flows"])
    assert result.exit_code != 0
