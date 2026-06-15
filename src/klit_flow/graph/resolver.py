"""Graph resolver — turns raw Symbol records into typed GraphNode/GraphEdge objects.

Four passes:
  1. Build File/Symbol nodes + DECLARES edges (File→Symbol, Class→Method).
  2. IMPORTS (exact): import path matched against an in-project FQN index.
  3. CALLS (best-effort): call-site name matched to declared symbol; confidence
     reflects how certain the match is (same-file > imported-file > any-file).
  4. EXTENDS / IMPLEMENTS: delegation_specifier matched against known types.

Parse failures in passes 3–4 are logged and skipped — never fatal.
"""

import hashlib
import logging
import re
from pathlib import Path

from tree_sitter import Node, Parser, Query, QueryCursor

from klit_flow.graph.model import GraphEdge, GraphNode, NodeKind, RelationType
from klit_flow.parsing.extractor import Symbol
from klit_flow.parsing.registry import get_ts_language
from klit_flow.walker import SourceFile

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ID helpers (must match extractor._make_id for symbol IDs)
# ─────────────────────────────────────────────────────────────────────────────

_SYMBOL_KIND_TO_NODE_KIND: dict[str, NodeKind] = {
    "Class": NodeKind.Class,
    "Interface": NodeKind.Interface,
    "Function": NodeKind.Function,
    "Method": NodeKind.Method,
}


def _file_id(file_path: str) -> str:
    return hashlib.sha256(f"File:{file_path}".encode()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# Package extraction (Kotlin — simple regex, no full parse needed)
# ─────────────────────────────────────────────────────────────────────────────

_PACKAGE_RE = re.compile(rb"^\s*package\s+([\w.]+)", re.MULTILINE)


def _extract_package(source: bytes) -> str:
    m = _PACKAGE_RE.search(source)
    return m.group(1).decode("utf-8") if m else ""


# ─────────────────────────────────────────────────────────────────────────────
# Tree-sitter helpers (re-used in passes 3 and 4)
# ─────────────────────────────────────────────────────────────────────────────

_KOTLIN_CALL_QUERY = "(call_expression) @call"
_KOTLIN_CLASS_QUERY = "(class_declaration) @class"


def _parse(file_path: Path, language: str) -> tuple | None:
    """Return (ts_lang, tree, source_bytes) or None on any error."""
    try:
        ts_lang = get_ts_language(language)
        source = file_path.read_bytes()
        return ts_lang, Parser(ts_lang).parse(source), source
    except Exception:
        logger.exception("Could not parse %s", file_path)
        return None


def _kt_callee_name(call_node: Node) -> str | None:
    """Extract the function name from a call_expression node.

    Handles both direct calls (``formatDate(...)``) and navigation calls
    (``activity.login(...)``).
    """
    for child in call_node.children:
        if child.type == "simple_identifier":
            return child.text.decode("utf-8")
        if child.type == "navigation_expression":
            return _last_simple_identifier(child)
    return None


def _last_simple_identifier(node: Node) -> str | None:
    """Return the text of the last simple_identifier in a subtree (DFS)."""
    result: list[str] = []

    def _walk(n: Node) -> None:
        if n.type == "simple_identifier":
            result.append(n.text.decode("utf-8"))
        for c in n.children:
            _walk(c)

    _walk(node)
    return result[-1] if result else None


def _kt_parent_names(class_node: Node) -> list[str]:
    """Return type names from delegation_specifiers of a class_declaration."""
    parents: list[str] = []
    for child in class_node.children:
        if child.type == "delegation_specifier":
            _collect_user_type_name(child, parents)
    return parents


def _collect_user_type_name(node: Node, out: list[str]) -> None:
    """Recurse into a delegation_specifier and collect base-type names.

    Only takes the first type_identifier of each user_type (skips generics).
    """
    if node.type == "user_type":
        for c in node.children:
            if c.type == "type_identifier":
                out.append(c.text.decode("utf-8"))
                return  # first identifier only — skip generic parameters
    for child in node.children:
        _collect_user_type_name(child, out)


# ─────────────────────────────────────────────────────────────────────────────
# Node builder
# ─────────────────────────────────────────────────────────────────────────────


def _sym_to_node(sym: Symbol) -> GraphNode | None:
    kind = _SYMBOL_KIND_TO_NODE_KIND.get(sym.kind)
    if kind is None:
        return None
    return GraphNode(
        id=sym.id,
        kind=kind,
        name=sym.name,
        file_path=sym.file_path,
        start_line=sym.start_line,
        end_line=sym.end_line,
        language=sym.language,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main resolver
# ─────────────────────────────────────────────────────────────────────────────


def resolve(
    source_files: list[SourceFile],
    symbols_by_file: dict[str, list[Symbol]],
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Resolve symbols into a complete set of GraphNode / GraphEdge objects."""

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    # ── Pass 1a: File nodes ───────────────────────────────────────────────────
    file_nodes: dict[str, GraphNode] = {}
    for sf in source_files:
        fp = str(sf.path)
        fn = GraphNode(
            id=_file_id(fp),
            kind=NodeKind.File,
            name=sf.path.name,
            file_path=fp,
            start_line=1,
            end_line=0,
            language=sf.language,
        )
        file_nodes[fp] = fn
        nodes.append(fn)

    # ── Pass 1b: Symbol nodes + DECLARES edges ────────────────────────────────
    # Indexes built during this pass and reused in later passes.
    sym_node_by_id: dict[str, GraphNode] = {}
    class_sym_by_name: dict[str, Symbol] = {}  # class/iface name → Symbol
    all_sym_by_name: dict[str, list[Symbol]] = {}  # name → [Symbol, …] (for CALLS)

    for fp, symbols in symbols_by_file.items():
        file_node = file_nodes.get(fp)

        # Collect class/interface symbols first (needed for method containment check)
        containers: list[Symbol] = [s for s in symbols if s.kind in ("Class", "Interface")]
        for s in containers:
            class_sym_by_name[s.name] = s

        for sym in symbols:
            gn = _sym_to_node(sym)
            if gn is None:
                continue  # Import symbols — not graph nodes

            sym_node_by_id[gn.id] = gn
            all_sym_by_name.setdefault(sym.name, []).append(sym)
            nodes.append(gn)

            # File DECLARES every top-level symbol
            if file_node:
                edges.append(
                    GraphEdge(
                        src_id=file_node.id,
                        dst_id=gn.id,
                        type=RelationType.DECLARES,
                        file_path=fp,
                        line=sym.start_line,
                    )
                )

            # Class/Interface DECLARES its methods (nested DECLARES)
            if sym.kind == "Method":
                for container in containers:
                    if container.start_line <= sym.start_line <= container.end_line:
                        container_gn = _sym_to_node(container)
                        if container_gn:
                            edges.append(
                                GraphEdge(
                                    src_id=container_gn.id,
                                    dst_id=gn.id,
                                    type=RelationType.DECLARES,
                                    file_path=fp,
                                    line=sym.start_line,
                                )
                            )
                        break

    # ── Pass 2: IMPORTS (exact) ───────────────────────────────────────────────
    # Build FQN → file_path index using package headers.
    fqn_to_file: dict[str, str] = {}
    for sf in source_files:
        fp = str(sf.path)
        syms = symbols_by_file.get(fp, [])
        if sf.language == "kotlin":
            try:
                source = sf.path.read_bytes()
                pkg = _extract_package(source)
            except OSError:
                pkg = ""
        else:
            pkg = ""
        for sym in syms:
            if sym.kind in ("Class", "Interface", "Function"):
                fqn = f"{pkg}.{sym.name}" if pkg else sym.name
                fqn_to_file[fqn] = fp

    seen_import_pairs: set[tuple[str, str]] = set()
    imported_by: dict[str, set[str]] = {}  # caller_fp → {callee_fp}

    for fp, symbols in symbols_by_file.items():
        file_node = file_nodes.get(fp)
        if not file_node:
            continue
        for sym in symbols:
            if sym.kind != "Import" or not sym.import_path:
                continue
            target_fp = fqn_to_file.get(sym.import_path)
            if not target_fp or target_fp == fp:
                continue  # external or self — skip
            target_fn = file_nodes.get(target_fp)
            if not target_fn:
                continue
            pair = (file_node.id, target_fn.id)
            if pair in seen_import_pairs:
                continue
            seen_import_pairs.add(pair)
            edges.append(
                GraphEdge(
                    src_id=file_node.id,
                    dst_id=target_fn.id,
                    type=RelationType.IMPORTS,
                    confidence=1.0,
                    file_path=fp,
                    line=sym.start_line,
                )
            )
            imported_by.setdefault(fp, set()).add(target_fp)

    # ── Pass 3: CALLS (best-effort) ───────────────────────────────────────────
    seen_call_pairs: set[tuple[str, str]] = set()

    for sf in source_files:
        fp = str(sf.path)
        if sf.language != "kotlin":
            continue  # extend per-phase as other languages are added
        parsed = _parse(sf.path, sf.language)
        if not parsed:
            continue
        ts_lang, tree, _ = parsed

        try:
            caps = QueryCursor(Query(ts_lang, _KOTLIN_CALL_QUERY)).captures(tree.root_node)
        except Exception:
            logger.exception("Call query failed for %s", fp)
            continue

        caller_syms = [s for s in symbols_by_file.get(fp, []) if s.kind in ("Function", "Method")]
        file_node = file_nodes[fp]

        for call_node in caps.get("call", []):
            callee_name = _kt_callee_name(call_node)
            if not callee_name:
                continue

            call_line = call_node.start_point[0] + 1

            # Enclosing caller symbol (best match by line range)
            caller_sym = next(
                (s for s in caller_syms if s.start_line <= call_line <= s.end_line),
                None,
            )
            src_id = (
                _sym_to_node(caller_sym).id
                if caller_sym and _sym_to_node(caller_sym)
                else file_node.id
            )

            candidates = all_sym_by_name.get(callee_name, [])
            for callee_sym in candidates:
                if callee_sym.kind not in ("Function", "Method"):
                    continue
                callee_gn = _sym_to_node(callee_sym)
                if not callee_gn:
                    continue

                if callee_sym.file_path == fp:
                    confidence = 0.9
                elif callee_sym.file_path in imported_by.get(fp, set()):
                    confidence = 0.7
                else:
                    confidence = 0.5

                pair = (src_id, callee_gn.id)
                if pair in seen_call_pairs:
                    continue
                seen_call_pairs.add(pair)
                edges.append(
                    GraphEdge(
                        src_id=src_id,
                        dst_id=callee_gn.id,
                        type=RelationType.CALLS,
                        confidence=confidence,
                        file_path=fp,
                        line=call_line,
                    )
                )

    # ── Pass 4: EXTENDS / IMPLEMENTS ─────────────────────────────────────────
    for sf in source_files:
        fp = str(sf.path)
        if sf.language != "kotlin":
            continue
        parsed = _parse(sf.path, sf.language)
        if not parsed:
            continue
        ts_lang, tree, _ = parsed

        try:
            caps = QueryCursor(Query(ts_lang, _KOTLIN_CLASS_QUERY)).captures(tree.root_node)
        except Exception:
            logger.exception("Class query failed for %s", fp)
            continue

        for class_node in caps.get("class", []):
            child_name = next(
                (
                    c.text.decode("utf-8")
                    for c in class_node.children
                    if c.type == "type_identifier"
                ),
                None,
            )
            if not child_name:
                continue
            child_sym = class_sym_by_name.get(child_name)
            if not child_sym:
                continue
            child_gn = _sym_to_node(child_sym)
            if not child_gn:
                continue

            for parent_name in _kt_parent_names(class_node):
                parent_sym = class_sym_by_name.get(parent_name)
                if not parent_sym:
                    continue  # external type, skip
                parent_gn = _sym_to_node(parent_sym)
                if not parent_gn:
                    continue
                rel = (
                    RelationType.EXTENDS if parent_sym.kind == "Class" else RelationType.IMPLEMENTS
                )
                edges.append(
                    GraphEdge(
                        src_id=child_gn.id,
                        dst_id=parent_gn.id,
                        type=rel,
                        confidence=1.0,
                        file_path=fp,
                        line=child_sym.start_line,
                    )
                )

    return nodes, edges
