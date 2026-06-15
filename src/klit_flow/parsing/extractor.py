"""Tree-sitter–based symbol extractor.

Runs language-specific .scm queries against a source file and returns Symbol
records. Parse failures are logged and return an empty list — never fatal.

Symbol.kind values produced here:
  "Class", "Interface", "Function", "Method", "Import"

"Import" symbols are extractor-internal; Phase 3 resolver converts them to
IMPORTS edges and they do not become graph nodes.
"""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Node, Parser, Query, QueryCursor

from klit_flow.parsing.registry import get_ts_language

logger = logging.getLogger(__name__)

_QUERIES_DIR = Path(__file__).parent / "queries"

# Grandparent node types that indicate a function is a method, not top-level.
# (grandparent = the container declaration; parent = its body node)
_METHOD_CONTAINERS = frozenset(
    {
        "class_declaration",
        "object_declaration",
        "companion_object",
    }
)


@dataclass(frozen=True)
class Symbol:
    id: str
    kind: str  # "Class" | "Interface" | "Function" | "Method" | "Import"
    name: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    import_path: str | None = None  # set only for kind="Import"


def _make_id(kind: str, file_path: str, name: str) -> str:
    raw = f"{kind}:{file_path}:{name}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _node_name(node: Node) -> str | None:
    """Return the name text from a declaration node by scanning direct children.

    Works for both class/interface (name is type_identifier) and function
    declarations (name is simple_identifier) in this grammar version.
    """
    for child in node.children:
        if child.type in ("type_identifier", "simple_identifier"):
            return child.text.decode("utf-8")
    return None


def _is_interface(node: Node) -> bool:
    """True when a class_declaration represents an interface.

    In this grammar both classes and interfaces use class_declaration; the
    first keyword child is 'interface' for interfaces, 'class' for classes.
    """
    return any(c.type == "interface" for c in node.children)


def _is_method(node: Node) -> bool:
    """True when a function_declaration is nested inside a class-like body."""
    parent = node.parent  # e.g. class_body
    if parent is None:
        return False
    grandparent = parent.parent  # e.g. class_declaration
    if grandparent is None:
        return False
    return grandparent.type in _METHOD_CONTAINERS


def _import_path(node: Node) -> str:
    """Extract the dotted module path from an import_header node."""
    text = node.text.decode("utf-8").strip()
    if text.startswith("import "):
        path = text[7:].strip()
        # Strip alias: "import foo.Bar as Baz" → "foo.Bar"
        if " as " in path:
            path = path.split(" as ")[0].strip()
        return path
    return text


def extract(file_path: Path, language: str) -> list[Symbol]:
    """Parse *file_path* with tree-sitter and return all Symbol records.

    Returns an empty list on any parse or I/O error (never raises).
    """
    query_file = _QUERIES_DIR / f"{language}.scm"
    if not query_file.exists():
        logger.warning("No query file for %r — skipping %s", language, file_path)
        return []

    try:
        ts_lang = get_ts_language(language)
    except ValueError:
        logger.warning("Unsupported language %r — skipping %s", language, file_path)
        return []

    try:
        source = file_path.read_bytes()
    except OSError:
        logger.warning("Cannot read %s", file_path)
        return []

    try:
        parser = Parser(ts_lang)
        tree = parser.parse(source)
        query_text = query_file.read_text(encoding="utf-8")
        query = Query(ts_lang, query_text)
        captures: dict[str, list[Node]] = QueryCursor(query).captures(tree.root_node)
    except Exception:
        logger.exception("Tree-sitter parse/query failed for %s", file_path)
        return []

    fp = str(file_path)
    symbols: list[Symbol] = []

    for node in captures.get("class", []):
        name = _node_name(node)
        if not name:
            continue
        kind = "Interface" if _is_interface(node) else "Class"
        symbols.append(
            Symbol(
                id=_make_id(kind, fp, name),
                kind=kind,
                name=name,
                file_path=fp,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                language=language,
            )
        )

    for node in captures.get("function", []):
        name = _node_name(node)
        if not name:
            continue
        kind = "Method" if _is_method(node) else "Function"
        symbols.append(
            Symbol(
                id=_make_id(kind, fp, name),
                kind=kind,
                name=name,
                file_path=fp,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                language=language,
            )
        )

    for node in captures.get("import", []):
        path = _import_path(node)
        symbols.append(
            Symbol(
                id=_make_id("Import", fp, path),
                kind="Import",
                name=path,
                file_path=fp,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                language=language,
                import_path=path,
            )
        )

    return symbols
