"""Optional local NL summaries via Ollama.

This module is intentionally a thin wrapper around the ``ollama`` Python
client.  All inference runs on the local machine — no cloud API calls are
ever made.  The feature is disabled by default and activated only when the
``--summaries`` flag is passed to ``klit-flow analyze``.

If Ollama is not installed, not running, or returns an empty response the
call is a silent no-op: ``None`` is returned and the caller keeps the
deterministic body it already has.  Parse failures are logged at DEBUG
level and never propagate to the caller.

Usage::

    from klit_flow.summarize import Summarizer

    summarizer = Summarizer(model="llama3.2")
    text = summarizer(node)       # str | None
"""

from __future__ import annotations

import logging

from klit_flow.graph.model import GraphNode

logger = logging.getLogger(__name__)

# A small, fast model available via ``ollama pull llama3.2``.
DEFAULT_MODEL = "llama3.2"

_SYSTEM = (
    "You are a concise technical writer.  "
    "Describe code artifacts in 1-2 sentences.  "
    "Be specific about the role in the codebase.  "
    "Do not include preamble or markdown."
)


def _build_prompt(node: GraphNode) -> str:
    return (
        f"Describe the {node.kind} named '{node.name}' "
        f"(file: {node.file_path}, lines {node.start_line}-{node.end_line}) "
        f"in 1-2 sentences."
    )


class Summarizer:
    """Callable that generates a 1–2 sentence NL summary for a ``GraphNode``.

    Parameters
    ----------
    model:
        Ollama model tag.  Defaults to ``llama3.2``.

    Examples
    --------
    >>> s = Summarizer()
    >>> text = s(node)         # returns str or None
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model

    def __call__(self, node: GraphNode) -> str | None:
        """Return a 1–2 sentence description, or ``None`` if Ollama is absent."""
        try:
            import ollama  # local import — keeps the module importable without ollama
        except ImportError:
            logger.debug("ollama package not installed; skipping summary for %s", node.name)
            return None

        prompt = _build_prompt(node)
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                system=_SYSTEM,
                options={"temperature": 0.2, "num_predict": 80},
            )
            text: str = response.response.strip()
            return text or None
        except Exception as exc:
            logger.debug("Ollama unavailable for %s: %s", node.name, exc)
            return None
