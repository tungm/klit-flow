"""Maps klit-flow language names to tree-sitter grammars."""

import logging

from tree_sitter import Language
from tree_sitter_language_pack import get_language

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


def get_ts_language(language: str) -> Language:
    ts_name = _TS_GRAMMAR.get(language)
    if ts_name is None:
        raise ValueError(f"No tree-sitter grammar registered for language: {language!r}")
    return get_language(ts_name)


def supported_languages() -> frozenset[str]:
    return frozenset(_TS_GRAMMAR)
