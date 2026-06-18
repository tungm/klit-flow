"""Maps klit-flow language names to tree-sitter grammars."""

import logging

from tree_sitter import Language
from tree_sitter_language_pack import downloaded_languages, get_language

logger = logging.getLogger(__name__)

# klit-flow language name → tree-sitter grammar name
_TS_GRAMMAR: dict[str, str] = {
    "kotlin": "kotlin",
    "java": "java",
    "swift": "swift",
    "dart": "dart",
    "typescript": "typescript",
    "javascript": "javascript",
    "python": "python",
    "xml": "xml",
}

# All tree-sitter grammar names required by klit-flow
REQUIRED_PARSERS: frozenset[str] = frozenset(_TS_GRAMMAR.values())


def get_ts_language(language: str) -> Language:
    ts_name = _TS_GRAMMAR.get(language)
    if ts_name is None:
        raise ValueError(f"No tree-sitter grammar registered for language: {language!r}")
    cached = set(downloaded_languages())
    if ts_name not in cached:
        raise RuntimeError(
            f"Tree-sitter parser for {ts_name!r} is not cached locally. "
            "Run 'klit-flow download-parsers' once to fetch all required parsers."
        )
    return get_language(ts_name)


def supported_languages() -> frozenset[str]:
    return frozenset(_TS_GRAMMAR)
