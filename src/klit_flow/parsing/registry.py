"""Maps klit-flow language names to tree-sitter grammars."""

import logging
import os
import platform as _platform
from pathlib import Path

from tree_sitter import Language
from tree_sitter_language_pack import configure, downloaded_languages, get_language
from tree_sitter_language_pack.options import PackConfig

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

# If KLIT_FLOW_PARSER_CACHE_DIR is set, point the library at that directory
# instead of the default system cache.  The release archive stores parsers in
# per-platform subdirectories:
#
#   parsers/
#     macos-arm64/
#     macos-x86_64/
#     linux-x86_64/
#     linux-aarch64/
#     windows-x86_64/
#     windows-aarch64/
#
# Set KLIT_FLOW_PARSER_CACHE_DIR to the parsers/ root; the correct
# platform subdirectory is chosen automatically at runtime.
_PARSER_CACHE_ENV = "KLIT_FLOW_PARSER_CACHE_DIR"

_configured = False


def current_platform() -> str:
    """Return the tree-sitter-language-pack platform identifier for this machine."""
    system = _platform.system()
    machine = _platform.machine().lower()
    if system == "Darwin":
        return "macos-arm64" if machine in ("arm64", "aarch64") else "macos-x86_64"
    if system == "Linux":
        return "linux-aarch64" if machine in ("aarch64", "arm64") else "linux-x86_64"
    if system == "Windows":
        return "windows-aarch64" if machine in ("aarch64", "arm64") else "windows-x86_64"
    return f"{system.lower()}-{machine}"


def _ensure_configured() -> None:
    """Apply KLIT_FLOW_PARSER_CACHE_DIR once before any parser access."""
    global _configured
    if _configured:
        return
    _configured = True

    plat = current_platform()
    logger.warning("[klit-flow] detected platform: %s", plat)

    env = os.environ.get(_PARSER_CACHE_ENV, "").strip()
    logger.warning("[klit-flow] %s = %r", _PARSER_CACHE_ENV, env or "(not set)")

    if not env:
        logger.warning(
            "[klit-flow] KLIT_FLOW_PARSER_CACHE_DIR not set; using default system cache. "
            "Set it to the bundled parsers/ directory for offline use."
        )
        return

    base = Path(env).expanduser().resolve()
    logger.warning("[klit-flow] parser cache base resolved to: %s  exists=%s", base, base.is_dir())

    # Prefer <base>/<platform>/ (all-platforms release layout) over <base>/ directly.
    plat_dir = base / plat
    logger.warning(
        "[klit-flow] checking platform subdir: %s  exists=%s", plat_dir, plat_dir.is_dir()
    )

    chosen = plat_dir if plat_dir.is_dir() else base
    if chosen.is_dir():
        contents = [f.name for f in chosen.iterdir()]
        logger.warning("[klit-flow] using parser cache: %s  files=%s", chosen, contents)
        configure(PackConfig(cache_dir=str(chosen)))
    else:
        logger.warning(
            "[klit-flow] KLIT_FLOW_PARSER_CACHE_DIR %r resolved to %s which is not a directory; "
            "using default system cache",
            env,
            base,
        )


def get_ts_language(language: str) -> Language:
    _ensure_configured()
    ts_name = _TS_GRAMMAR.get(language)
    if ts_name is None:
        raise ValueError(f"No tree-sitter grammar registered for language: {language!r}")
    cached = set(downloaded_languages())
    if ts_name not in cached:
        raise RuntimeError(
            f"Tree-sitter parser for {ts_name!r} is not cached locally. "
            "Set KLIT_FLOW_PARSER_CACHE_DIR to the bundled parsers/ directory, "
            "or run 'klit-flow download-parsers' to fetch parsers from the network."
        )
    return get_language(ts_name)


def supported_languages() -> frozenset[str]:
    return frozenset(_TS_GRAMMAR)
