"""File discovery for klit-flow.

Recursively walks a source tree, respecting .gitignore and .klit-flowignore,
detecting language by file extension, and skipping oversized files.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import pathspec

logger = logging.getLogger(__name__)

# Supported extensions → canonical language name
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".swift": "swift",
    ".dart": "dart",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".java": "java",
    ".py": "python",
    ".xml": "xml",  # Android navigation graphs / layouts
}

DEFAULT_MAX_FILE_SIZE: int = 1 * 1024 * 1024  # 1 MiB


@dataclass(frozen=True)
class SourceFile:
    path: Path
    rel_path: str  # POSIX, relative to the walked root
    language: str
    size: int  # bytes


def _load_spec(path: Path) -> pathspec.PathSpec:
    """Return a PathSpec from a gitignore-style file, or empty spec if absent."""
    if path.is_file():
        lines = path.read_text(encoding="utf-8").splitlines()
        return pathspec.PathSpec.from_lines("gitignore", lines)
    return pathspec.PathSpec.from_lines("gitignore", [])


def walk(
    root: Path | str,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> list[SourceFile]:
    """Discover all supported source files under *root*.

    Skips files matched by ``root/.gitignore`` or ``root/.klit-flowignore``,
    files whose extension is not in EXTENSION_TO_LANGUAGE, and files larger
    than *max_file_size* bytes.

    Returns a stable-sorted list of SourceFile records.
    """
    root = Path(root).resolve()
    git_spec = _load_spec(root / ".gitignore")
    klit_spec = _load_spec(root / ".klit-flowignore")

    results: list[SourceFile] = []
    for path in sorted(root.rglob("*"), key=lambda p: p.as_posix()):
        if not path.is_file():
            continue

        rel = path.relative_to(root).as_posix()

        if git_spec.match_file(rel) or klit_spec.match_file(rel):
            logger.debug("ignored: %s", rel)
            continue

        lang = EXTENSION_TO_LANGUAGE.get(path.suffix.lower())
        if lang is None:
            continue

        size = path.stat().st_size
        if size > max_file_size:
            logger.warning("skipped (too large, %d bytes): %s", size, rel)
            continue

        results.append(SourceFile(path=path, rel_path=rel, language=lang, size=size))

    return results
