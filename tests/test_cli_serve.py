"""Tests for the `klit-flow serve` command surface.

These cover the CLI wiring (the ``--host`` option and the missing-index guard)
without actually starting the blocking MCP/web servers.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from klit_flow.cli import app

runner = CliRunner()


def test_serve_help_documents_host_option() -> None:
    result = runner.invoke(app, ["serve", "--help"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "--host" in result.output
    assert "--port" in result.output


def test_serve_help_mentions_docker_host_hint() -> None:
    result = runner.invoke(app, ["serve", "--help"], catch_exceptions=False)
    # The hint steers Docker users toward binding all interfaces.
    assert "0.0.0.0" in result.output


def test_serve_without_index_exits_nonzero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["serve"])
    assert result.exit_code != 0
    assert "No index found" in result.output
