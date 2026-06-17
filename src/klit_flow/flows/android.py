"""Android / Kotlin screen-flow extractor.

Screen detection
----------------
Classes whose name ends in ``Activity`` or ``Fragment`` are treated as screens.
This name-based heuristic covers the vast majority of Android apps without
requiring full type-resolution of ``AppCompatActivity`` / ``Fragment`` chains.

Navigation edge detection (two sources)
-----------------------------------------
1. **Code** — tree-sitter AST finds ``startActivity(Intent(ctx, Target::class.java))``
   calls.  Confidence 0.8.  All enclosing ``if``/``when`` control-flow nodes
   are walked to produce structured ``ConditionLevel`` objects.
   Trigger inferred from ancestors: ``setOnClickListener`` → ``button_tap``;
   ``when``-branch → ``api_response``; otherwise ``programmatic``.
   Explicit ``// klit:condition:`` annotations override auto-detection for
   cross-method chains.

   **Cross-file detection**: ``startActivity`` calls in non-screen files are
   traced backward through the call graph (up to 3 hops) to find the
   originating screen.  Conditions from the entire call chain are merged.

2. **XML nav graph** — ``res/navigation/*.xml`` ``<fragment>``/``<action>``
   elements parsed with stdlib ``xml.etree.ElementTree``.  Confidence 0.95.

Known limitations (confidence < 1 by design)
----------------------------------------------
- Deep links and string-built route names are not resolved.
- Dynamic ``startActivity`` calls (variable Intent) are not captured.
- ``navController.navigate(R.id.x)`` is not yet parsed from code.
"""

import hashlib
import logging
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Node, Parser, Query, QueryCursor

from klit_flow.flows.base import ScreenFlowExtractor
from klit_flow.graph.model import ConditionLevel, GraphEdge, GraphNode, NodeKind, RelationType
from klit_flow.parsing.extractor import Symbol
from klit_flow.parsing.registry import get_ts_language
from klit_flow.walker import SourceFile

logger = logging.getLogger(__name__)

# Heuristic: class names ending with these suffixes are considered screens.
_SCREEN_SUFFIXES: tuple[str, ...] = ("Activity", "Fragment")

# Explicit condition annotation: // klit:condition: cond1, cond2, ...
_KLIT_CONDITION_RE = re.compile(r"//\s*klit:condition:\s*(.+)", re.IGNORECASE)

# Android XML namespace URIs
_NS_ANDROID = "http://schemas.android.com/apk/res/android"
_NS_APP = "http://schemas.android.com/apk/res-auto"

# Tree-sitter query for call expressions
_KOTLIN_CALL_QUERY = "(call_expression) @call"

# Maximum hops for cross-file call-graph trace
_MAX_TRACE_DEPTH = 3


@dataclass
class _NavCall:
    """A ``startActivity`` call found in any Kotlin file."""

    file_path: str
    dest_screen_name: str
    conditions: list[ConditionLevel]
    trigger: str
    line: int
    enclosing_method: str | None
    enclosing_class: str | None


@dataclass
class _CallSite:
    """A call to a method found in any Kotlin file."""

    file_path: str
    callee_name: str
    conditions: list[ConditionLevel]
    line: int
    enclosing_method: str | None
    enclosing_class: str | None
    in_click_listener: bool = False


def _screen_id(class_name: str, file_path: str) -> str:
    return hashlib.sha256(f"Screen:{file_path}:{class_name}".encode()).hexdigest()[:16]


def _is_screen(class_name: str) -> bool:
    return class_name.endswith(_SCREEN_SUFFIXES)


def _strip_id_ref(ref: str) -> str:
    """``@+id/foo`` or ``@id/foo`` → ``foo``."""
    if ref.startswith("@+id/"):
        return ref[5:]
    if ref.startswith("@id/"):
        return ref[4:]
    return ref


# ── Tree-sitter helpers ──────────────────────────────────────────────────────


def _find_descendant(node: Node, target_type: str) -> Node | None:
    """Find first descendant of the given type (DFS)."""
    if node.type == target_type:
        return node
    for child in node.children:
        result = _find_descendant(child, target_type)
        if result:
            return result
    return None


def _callee_name(call_node: Node) -> str | None:
    """Get the function name from a call_expression (handles navigation)."""
    for child in call_node.children:
        if child.type == "simple_identifier":
            return child.text.decode("utf-8")
        if child.type == "navigation_expression":
            last_id: str | None = None
            for c in child.children:
                if c.type == "simple_identifier":
                    last_id = c.text.decode("utf-8")
                elif c.type == "navigation_suffix":
                    for sc in c.children:
                        if sc.type == "simple_identifier":
                            last_id = sc.text.decode("utf-8")
            return last_id
    return None


def _extract_intent_target(start_activity_call: Node) -> str | None:
    """Extract ClassName from ``startActivity(Intent(ctx, ClassName::class.java))``."""
    call_suffix = _find_descendant(start_activity_call, "call_suffix")
    if call_suffix is None:
        return None
    ref = _find_descendant(call_suffix, "callable_reference")
    if ref is None:
        return None
    type_id = _find_descendant(ref, "type_identifier")
    return type_id.text.decode("utf-8") if type_id else None


def _extract_if_condition_text(if_node: Node) -> str | None:
    """Extract the condition expression text from an ``if_expression``."""
    in_parens = False
    for child in if_node.children:
        if child.type == "(":
            in_parens = True
            continue
        if child.type == ")":
            break
        if in_parens:
            text = child.text.decode("utf-8").strip()
            return text[:120] if text else None
    return None


def _is_in_else_branch(if_node: Node, target_byte: int) -> bool:
    """Check if *target_byte* falls in the else branch of an ``if_expression``."""
    for child in if_node.children:
        if child.type == "else" and target_byte > child.start_byte:
            return True
    return False


def _extract_when_entry_condition(when_entry: Node) -> tuple[str | None, str]:
    """Extract condition from a ``when_entry``, combining with ``when_subject``.

    Returns ``(expression, kind)`` where *kind* is ``"when_branch"`` or
    ``"when_else"``.
    """
    # Get the when_expression parent to find the subject
    when_expr = when_entry.parent
    subject_text = ""
    if when_expr and when_expr.type == "when_expression":
        subject_node = _find_descendant(when_expr, "when_subject")
        if subject_node:
            for child in subject_node.children:
                if child.type not in ("(", ")"):
                    subject_text = child.text.decode("utf-8")
                    break

    # Check for else branch (default)
    cond_node = _find_descendant(when_entry, "when_condition")
    if cond_node is None:
        for child in when_entry.children:
            if child.text == b"else":
                expr = f"{subject_text} (default)" if subject_text else "(default)"
                return expr, "when_else"
        return None, "when_branch"

    # Type test: ``is SomeType``
    type_test = _find_descendant(cond_node, "type_test")
    if type_test:
        user_type = _find_descendant(type_test, "user_type")
        if user_type:
            type_text = user_type.text.decode("utf-8")
            if subject_text:
                return f"{subject_text} is {type_text}", "when_branch"
            return f"is {type_text}", "when_branch"

    # Value-based condition
    cond_text = cond_node.text.decode("utf-8").strip()
    if cond_text:
        if subject_text:
            return f"{subject_text} == {cond_text}", "when_branch"
        return cond_text, "when_branch"
    return None, "when_branch"


def _collect_ancestor_conditions(node: Node) -> list[ConditionLevel]:
    """Walk up from *node* collecting all enclosing ``if``/``when`` conditions.

    Returns conditions from outermost to innermost.
    """
    conditions: list[ConditionLevel] = []
    target_byte = node.start_byte
    current = node.parent

    while current:
        if current.type == "if_expression":
            cond_text = _extract_if_condition_text(current)
            if cond_text:
                is_else = _is_in_else_branch(current, target_byte)
                conditions.append(
                    ConditionLevel(
                        expression=cond_text,
                        kind="else" if is_else else "if",
                        source_line=current.start_point[0] + 1,
                    )
                )
        elif current.type == "when_entry":
            expr, kind = _extract_when_entry_condition(current)
            if expr:
                conditions.append(
                    ConditionLevel(
                        expression=expr,
                        kind=kind,
                        source_line=current.start_point[0] + 1,
                    )
                )
        current = current.parent

    conditions.reverse()  # outermost first
    return conditions


def _has_click_listener_ancestor(node: Node) -> bool:
    """Check if *node* is inside a ``setOnClickListener`` lambda."""
    current = node.parent
    while current:
        if current.type == "call_expression":
            name = _callee_name(current)
            if name == "setOnClickListener":
                return True
        current = current.parent
    return False


def _parse_annotation_conditions(annotation_text: str) -> list[ConditionLevel]:
    """Parse a ``// klit:condition:`` annotation into ``ConditionLevel`` objects."""
    parts = [p.strip() for p in annotation_text.split(",") if p.strip()]
    return [ConditionLevel(expression=p, kind="annotation") for p in parts]


def _enclosing_function_name(node: Node) -> str | None:
    """Walk up from *node* to find the enclosing ``function_declaration`` name."""
    current = node.parent
    while current:
        if current.type == "function_declaration":
            for child in current.children:
                if child.type == "simple_identifier":
                    return child.text.decode("utf-8")
        current = current.parent
    return None


def _enclosing_class_name(node: Node) -> str | None:
    """Walk up from *node* to find the enclosing ``class_declaration`` name."""
    current = node.parent
    while current:
        if current.type == "class_declaration":
            type_id = _find_descendant(current, "type_identifier")
            if type_id:
                return type_id.text.decode("utf-8")
        current = current.parent
    return None


class AndroidFlowExtractor(ScreenFlowExtractor):
    # ── Screen detection ──────────────────────────────────────────────────────

    def extract_screens(
        self,
        source_files: list[SourceFile],
        symbols_by_file: dict[str, list[Symbol]],
    ) -> list[GraphNode]:
        screens: list[GraphNode] = []
        for sf in source_files:
            fp = str(sf.path)
            for sym in symbols_by_file.get(fp, []):
                if sym.kind == "Class" and _is_screen(sym.name):
                    screens.append(
                        GraphNode(
                            id=_screen_id(sym.name, fp),
                            kind=NodeKind.Screen,
                            name=sym.name,
                            file_path=fp,
                            start_line=sym.start_line,
                            end_line=sym.end_line,
                            language=sym.language,
                        )
                    )
        return screens

    # ── Flow (NAVIGATES_TO) detection ─────────────────────────────────────────

    def extract_flows(
        self,
        source_files: list[SourceFile],
        symbols_by_file: dict[str, list[Symbol]],
        screen_nodes: list[GraphNode],
    ) -> list[GraphEdge]:
        screen_by_name: dict[str, GraphNode] = {n.name: n for n in screen_nodes}
        raw: list[GraphEdge] = []

        # Source 1: code-based detection
        raw.extend(self._from_code(source_files, symbols_by_file, screen_by_name))

        # Source 2: XML nav graph
        raw.extend(self._from_xml(source_files, screen_by_name))

        # De-duplicate: keep highest-confidence edge per (src, dst) pair.
        best: dict[tuple[str, str], GraphEdge] = {}
        for edge in raw:
            key = (edge.src_id, edge.dst_id)
            if key not in best:
                best[key] = edge
                continue
            existing = best[key]
            if edge.confidence >= existing.confidence:
                trigger = edge.trigger
                if trigger == "programmatic" and existing.trigger != "programmatic":
                    trigger = existing.trigger
                conditions = edge.conditions or existing.conditions
                best[key] = edge.model_copy(update={"trigger": trigger, "conditions": conditions})
            elif existing.trigger == "programmatic" and edge.trigger != "programmatic":
                conditions = existing.conditions or edge.conditions
                best[key] = existing.model_copy(
                    update={"trigger": edge.trigger, "conditions": conditions}
                )
            elif not existing.conditions and edge.conditions:
                best[key] = existing.model_copy(update={"conditions": edge.conditions})
        return list(best.values())

    # ── Private helpers ───────────────────────────────────────────────────────

    def _from_code(
        self,
        source_files: list[SourceFile],
        symbols_by_file: dict[str, list[Symbol]],
        screen_by_name: dict[str, GraphNode],
    ) -> list[GraphEdge]:
        # Build file→screen index
        screen_by_file: dict[str, GraphNode] = {}
        for sf in source_files:
            fp = str(sf.path)
            for sym in symbols_by_file.get(fp, []):
                if sym.kind == "Class" and sym.name in screen_by_name:
                    screen_by_file[fp] = screen_by_name[sym.name]
                    break

        # Phase 1: scan ALL Kotlin files for startActivity calls and general call sites
        nav_calls: list[_NavCall] = []
        call_index: dict[str, list[_CallSite]] = defaultdict(list)

        for sf in source_files:
            if sf.language != "kotlin":
                continue
            fp = str(sf.path)

            try:
                source_bytes = sf.path.read_bytes()
            except OSError:
                logger.warning("Cannot read %s", fp)
                continue

            try:
                ts_lang = get_ts_language("kotlin")
                parser = Parser(ts_lang)
                tree = parser.parse(source_bytes)
                query = Query(ts_lang, _KOTLIN_CALL_QUERY)
                caps = QueryCursor(query).captures(tree.root_node)
            except Exception:
                logger.exception("Tree-sitter parse failed for %s", fp)
                continue

            for call_node in caps.get("call", []):
                name = _callee_name(call_node)
                if not name:
                    continue

                enc_method = _enclosing_function_name(call_node)
                enc_class = _enclosing_class_name(call_node)

                if name == "startActivity":
                    dest_name = _extract_intent_target(call_node)
                    if dest_name is None:
                        continue

                    # Gather conditions and trigger
                    ctx_start = max(0, call_node.start_byte - 300)
                    close_ctx = source_bytes[ctx_start : call_node.end_byte].decode(
                        "utf-8", errors="replace"
                    )
                    ann_m = _KLIT_CONDITION_RE.search(close_ctx)

                    if ann_m:
                        conditions = _parse_annotation_conditions(ann_m.group(1))
                        trigger = "api_response"
                    else:
                        conditions = _collect_ancestor_conditions(call_node)
                        has_when = any(c.kind in ("when_branch", "when_else") for c in conditions)
                        is_click = _has_click_listener_ancestor(call_node)
                        if has_when:
                            trigger = "api_response"
                        elif is_click:
                            trigger = "button_tap"
                        else:
                            trigger = "programmatic"

                    line = source_bytes[: call_node.start_byte].count(b"\n") + 1
                    nav_calls.append(
                        _NavCall(
                            file_path=fp,
                            dest_screen_name=dest_name,
                            conditions=conditions,
                            trigger=trigger,
                            line=line,
                            enclosing_method=enc_method,
                            enclosing_class=enc_class,
                        )
                    )
                else:
                    # Index all other calls for cross-file tracing
                    conditions = _collect_ancestor_conditions(call_node)
                    is_click = _has_click_listener_ancestor(call_node)
                    line = source_bytes[: call_node.start_byte].count(b"\n") + 1
                    call_index[name].append(
                        _CallSite(
                            file_path=fp,
                            callee_name=name,
                            conditions=conditions,
                            line=line,
                            enclosing_method=enc_method,
                            enclosing_class=enc_class,
                            in_click_listener=is_click,
                        )
                    )

        # Phase 2: direct edges for startActivity calls in screen files
        edges: list[GraphEdge] = []
        indirect_nav_calls: list[_NavCall] = []

        for nc in nav_calls:
            src_screen = screen_by_file.get(nc.file_path)
            dst_screen = screen_by_name.get(nc.dest_screen_name)
            if dst_screen is None:
                continue

            if src_screen is not None and src_screen.id != dst_screen.id:
                edges.append(
                    GraphEdge(
                        src_id=src_screen.id,
                        dst_id=dst_screen.id,
                        type=RelationType.NAVIGATES_TO,
                        confidence=0.8,
                        file_path=nc.file_path,
                        line=nc.line,
                        trigger=nc.trigger,
                        conditions=nc.conditions,
                    )
                )
            elif src_screen is None:
                indirect_nav_calls.append(nc)

        # Phase 3: cross-file trace for startActivity calls in non-screen files
        for nc in indirect_nav_calls:
            dst_screen = screen_by_name.get(nc.dest_screen_name)
            if dst_screen is None:
                continue

            origins = _trace_to_screens(nc, call_index, screen_by_file, screen_by_name)
            for src_screen, chain_conditions, chain_trigger in origins:
                if src_screen.id == dst_screen.id:
                    continue
                # Merge: caller conditions first, then callee conditions
                merged = chain_conditions + nc.conditions
                trigger = chain_trigger if chain_trigger != "programmatic" else nc.trigger
                edges.append(
                    GraphEdge(
                        src_id=src_screen.id,
                        dst_id=dst_screen.id,
                        type=RelationType.NAVIGATES_TO,
                        confidence=0.7,  # lower confidence for cross-file
                        file_path=nc.file_path,
                        line=nc.line,
                        trigger=trigger,
                        conditions=merged,
                    )
                )

        return edges

    def _from_xml(
        self,
        source_files: list[SourceFile],
        screen_by_name: dict[str, GraphNode],
    ) -> list[GraphEdge]:
        edges: list[GraphEdge] = []
        for sf in source_files:
            if sf.language != "xml" or sf.path.parent.name != "navigation":
                continue
            edges.extend(_parse_nav_xml(sf.path, screen_by_name))
        return edges


def _trace_to_screens(
    nav_call: _NavCall,
    call_index: dict[str, list[_CallSite]],
    screen_by_file: dict[str, GraphNode],
    screen_by_name: dict[str, GraphNode],
) -> list[tuple[GraphNode, list[ConditionLevel], str]]:
    """BFS backward through the call graph to find originating screen(s).

    Returns a list of ``(screen_node, chain_conditions, trigger)`` tuples.
    *chain_conditions* contains conditions from the call chain (outermost
    caller first), NOT including the nav_call's own conditions.
    """
    results: list[tuple[GraphNode, list[ConditionLevel], str]] = []
    # BFS queue: (method_name_to_find, accumulated_conditions, trigger, depth, visited)
    target_method = nav_call.enclosing_method
    if target_method is None:
        return results

    queue: list[tuple[str, list[ConditionLevel], str, int, set[str]]] = [
        (target_method, [], "programmatic", 0, set()),
    ]

    while queue:
        method_name, acc_conds, acc_trigger, depth, visited = queue.pop(0)
        if depth >= _MAX_TRACE_DEPTH:
            continue

        for site in call_index.get(method_name, []):
            site_key = f"{site.file_path}:{site.line}"
            if site_key in visited:
                continue
            new_visited = visited | {site_key}

            # Merge conditions: site conditions + accumulated so far
            merged_conds = site.conditions + acc_conds
            trigger = acc_trigger
            if site.in_click_listener:
                trigger = "button_tap"
            elif any(c.kind in ("when_branch", "when_else") for c in site.conditions):
                trigger = "api_response"

            # Check if this call site is in a screen file
            src_screen = screen_by_file.get(site.file_path)
            if src_screen is not None:
                results.append((src_screen, merged_conds, trigger))
            elif site.enclosing_method:
                queue.append((site.enclosing_method, merged_conds, trigger, depth + 1, new_visited))

    return results


def _parse_nav_xml(xml_path: Path, screen_by_name: dict[str, GraphNode]) -> list[GraphEdge]:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        logger.warning("Could not parse nav XML: %s", xml_path)
        return []

    # Build: fragment_id_str → simple_class_name
    id_to_class: dict[str, str] = {}
    for frag in root.iter("fragment"):
        frag_id = _strip_id_ref(frag.get(f"{{{_NS_ANDROID}}}id", ""))
        android_name = frag.get(f"{{{_NS_ANDROID}}}name", "")
        simple_name = android_name.rsplit(".", 1)[-1] if android_name else ""
        if frag_id and simple_name:
            id_to_class[frag_id] = simple_name

    edges: list[GraphEdge] = []
    fp = str(xml_path)

    for frag in root.iter("fragment"):
        src_frag_id = _strip_id_ref(frag.get(f"{{{_NS_ANDROID}}}id", ""))
        src_class = id_to_class.get(src_frag_id)
        src_screen = screen_by_name.get(src_class) if src_class else None
        if src_screen is None:
            continue

        for action in frag.iter("action"):
            dest_ref = _strip_id_ref(action.get(f"{{{_NS_APP}}}destination", ""))
            dst_class = id_to_class.get(dest_ref)
            dst_screen = screen_by_name.get(dst_class) if dst_class else None
            if dst_screen is None or dst_screen.id == src_screen.id:
                continue

            edges.append(
                GraphEdge(
                    src_id=src_screen.id,
                    dst_id=dst_screen.id,
                    type=RelationType.NAVIGATES_TO,
                    confidence=0.95,
                    file_path=fp,
                    trigger="programmatic",
                )
            )
    return edges
