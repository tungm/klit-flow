"""Phase 9 acceptance tests: optional local NL summaries.

Tests are split into three groups:

1. **Summarizer unit tests** — verify the ``Summarizer`` callable behaves
   correctly: returns text when Ollama responds, returns ``None`` when Ollama
   is absent or returns an empty string, never raises.

2. **Emitter integration tests** — verify that ``emit_module_docs`` and
   ``emit_screen_docs`` inject the summary into the Markdown body when a
   summarizer is provided, and are unchanged when it is not.

3. **Off-by-default / backwards-compat tests** — verify that omitting
   ``summarizer`` (default ``None``) produces identical output to the
   pre-Phase-9 baseline.

No real Ollama process is started.  ``ollama.generate`` is monkeypatched in
every test that exercises the live path.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import frontmatter
import pytest

from klit_flow.emit.markdown_emitter import emit_module_docs, emit_screen_docs
from klit_flow.graph.model import GraphNode, NodeKind
from klit_flow.summarize import DEFAULT_MODEL, Summarizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _file_node(node_id: str = "f1", name: str = "Main.kt") -> GraphNode:
    return GraphNode(
        id=node_id,
        kind=NodeKind.File,
        name=name,
        file_path=f"/src/{name}",
        start_line=1,
        end_line=100,
        language="kotlin",
    )


def _screen_node(node_id: str = "s1", name: str = "MainActivity") -> GraphNode:
    return GraphNode(
        id=node_id,
        kind=NodeKind.Screen,
        name=name,
        file_path=f"/src/{name}.kt",
        start_line=1,
        end_line=80,
        language="kotlin",
    )


def _fake_response(text: str) -> SimpleNamespace:
    """Simulate an ``ollama.GenerateResponse``-like object."""
    return SimpleNamespace(response=text)


# ---------------------------------------------------------------------------
# Summarizer unit tests
# ---------------------------------------------------------------------------


def test_summarizer_returns_text_when_ollama_responds() -> None:
    with patch("ollama.generate", return_value=_fake_response("Handles user auth.")):
        s = Summarizer(model="llama3.2")
        result = s(_file_node())
    assert result == "Handles user auth."


def test_summarizer_returns_none_when_ollama_connection_error() -> None:
    import ollama

    with patch("ollama.generate", side_effect=ollama.ResponseError("connection refused")):
        s = Summarizer()
        result = s(_file_node())
    assert result is None


def test_summarizer_returns_none_when_generic_exception() -> None:
    with patch("ollama.generate", side_effect=RuntimeError("boom")):
        s = Summarizer()
        result = s(_file_node())
    assert result is None


def test_summarizer_returns_none_for_empty_response() -> None:
    with patch("ollama.generate", return_value=_fake_response("   ")):
        s = Summarizer()
        result = s(_file_node())
    assert result is None


def test_summarizer_strips_whitespace() -> None:
    with patch("ollama.generate", return_value=_fake_response("  Entry point.  \n")):
        s = Summarizer()
        result = s(_file_node())
    assert result == "Entry point."


def test_summarizer_passes_model_to_ollama() -> None:
    mock = MagicMock(return_value=_fake_response("ok"))
    with patch("ollama.generate", mock):
        Summarizer(model="mistral")(_file_node())
    assert mock.call_args.kwargs["model"] == "mistral"


def test_summarizer_default_model() -> None:
    assert Summarizer().model == DEFAULT_MODEL


def test_summarizer_never_raises_on_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the ollama package is not installed at all, Summarizer must return None."""
    import sys

    monkeypatch.setitem(sys.modules, "ollama", None)  # simulate missing package
    s = Summarizer()
    result = s(_file_node())
    assert result is None


# ---------------------------------------------------------------------------
# Emitter integration: module docs
# ---------------------------------------------------------------------------


def test_emit_module_docs_with_summarizer_adds_summary(tmp_path: Path) -> None:
    node = _file_node("f1", "Auth.kt")
    summarizer = lambda n: "Handles authentication logic."  # noqa: E731
    paths = emit_module_docs([node], [], tmp_path, summarizer=summarizer)
    assert len(paths) == 1
    content = paths[0].read_text()
    assert "Handles authentication logic." in content


def test_emit_module_docs_with_summarizer_adds_summary_section(tmp_path: Path) -> None:
    node = _file_node("f1", "Auth.kt")
    paths = emit_module_docs([node], [], tmp_path, summarizer=lambda n: "Auth module.")
    content = paths[0].read_text()
    assert "### Summary" in content


def test_emit_module_docs_without_summarizer_unchanged(tmp_path: Path) -> None:
    """Default (no summarizer) must not add a Summary section."""
    node = _file_node("f1", "Auth.kt")
    paths = emit_module_docs([node], [], tmp_path)
    content = paths[0].read_text()
    assert "### Summary" not in content


def test_emit_module_docs_summarizer_returns_none_no_section(tmp_path: Path) -> None:
    """When summarizer returns None (Ollama absent), no Summary section is added."""
    node = _file_node("f1", "Auth.kt")
    paths = emit_module_docs([node], [], tmp_path, summarizer=lambda n: None)
    content = paths[0].read_text()
    assert "### Summary" not in content


def test_emit_module_docs_frontmatter_unchanged_with_summarizer(tmp_path: Path) -> None:
    """Adding a summary must not corrupt the YAML frontmatter."""
    node = _file_node("f1", "Auth.kt")
    paths = emit_module_docs([node], [], tmp_path, summarizer=lambda n: "Auth module.")
    post = frontmatter.load(str(paths[0]))
    assert post["id"] == "f1"
    assert post["kind"] == "Module"


# ---------------------------------------------------------------------------
# Emitter integration: screen docs
# ---------------------------------------------------------------------------


def test_emit_screen_docs_with_summarizer_adds_summary(tmp_path: Path) -> None:
    screen = _screen_node("s1", "LoginActivity")
    paths = emit_screen_docs([screen], [], tmp_path, summarizer=lambda n: "Login screen.")
    assert len(paths) == 1
    content = paths[0].read_text()
    assert "Login screen." in content


def test_emit_screen_docs_with_summarizer_adds_summary_section(tmp_path: Path) -> None:
    screen = _screen_node("s1", "LoginActivity")
    paths = emit_screen_docs([screen], [], tmp_path, summarizer=lambda n: "Login screen.")
    assert "### Summary" in paths[0].read_text()


def test_emit_screen_docs_without_summarizer_unchanged(tmp_path: Path) -> None:
    screen = _screen_node("s1", "LoginActivity")
    paths = emit_screen_docs([screen], [], tmp_path)
    assert "### Summary" not in paths[0].read_text()


def test_emit_screen_docs_summarizer_returns_none_no_section(tmp_path: Path) -> None:
    screen = _screen_node("s1", "LoginActivity")
    paths = emit_screen_docs([screen], [], tmp_path, summarizer=lambda n: None)
    assert "### Summary" not in paths[0].read_text()


def test_emit_screen_docs_frontmatter_unchanged_with_summarizer(tmp_path: Path) -> None:
    screen = _screen_node("s1", "LoginActivity")
    paths = emit_screen_docs([screen], [], tmp_path, summarizer=lambda n: "Login.")
    post = frontmatter.load(str(paths[0]))
    assert post["id"] == "s1"
    assert post["kind"] == "Screen"


# ---------------------------------------------------------------------------
# Off-by-default / backwards compatibility
# ---------------------------------------------------------------------------


def test_summarizer_none_produces_same_output_as_baseline(tmp_path: Path) -> None:
    """emit_module_docs(summarizer=None) == emit_module_docs() (no extra content)."""
    node = _file_node("f1", "Main.kt")
    p1 = emit_module_docs([node], [], tmp_path / "a")[0]
    p2 = emit_module_docs([node], [], tmp_path / "b", summarizer=None)[0]
    assert p1.read_text() == p2.read_text()


def test_ollama_absent_is_noop_for_emitter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If Ollama is not running, the doc is written with its deterministic body only."""
    import sys

    monkeypatch.setitem(sys.modules, "ollama", None)
    s = Summarizer()  # will return None silently
    node = _file_node("f1", "Main.kt")
    paths = emit_module_docs([node], [], tmp_path, summarizer=s)
    content = paths[0].read_text()
    assert "### Summary" not in content
    assert "## Main.kt" in content


def test_summarizer_with_live_ollama_patch(tmp_path: Path) -> None:
    """Simulate a successful Ollama call end-to-end through the emitter."""
    with patch("ollama.generate", return_value=_fake_response("Entry point of the app.")):
        s = Summarizer(model="llama3.2")
        node = _file_node("f1", "Main.kt")
        paths = emit_module_docs([node], [], tmp_path, summarizer=s)

    content = paths[0].read_text()
    assert "Entry point of the app." in content
    assert "### Summary" in content
