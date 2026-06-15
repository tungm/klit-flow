"""Android / Kotlin screen-flow extractor.

Screen detection
----------------
Classes whose name ends in ``Activity`` or ``Fragment`` are treated as screens.
This name-based heuristic covers the vast majority of Android apps without
requiring full type-resolution of ``AppCompatActivity`` / ``Fragment`` chains.

Navigation edge detection (two sources)
-----------------------------------------
1. **Code** — ``startActivity(Intent(ctx, TargetClass::class.java))`` regex.
   Confidence 0.8.  Trigger inferred from surrounding context:
   ``setOnClickListener`` → ``button_tap``; otherwise ``programmatic``.

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
from pathlib import Path

from klit_flow.flows.base import ScreenFlowExtractor
from klit_flow.graph.model import GraphEdge, GraphNode, NodeKind, RelationType
from klit_flow.parsing.extractor import Symbol
from klit_flow.walker import SourceFile

logger = logging.getLogger(__name__)

# Heuristic: class names ending with these suffixes are considered screens.
_SCREEN_SUFFIXES: tuple[str, ...] = ("Activity", "Fragment")

# Matches: startActivity(Intent(ctx, ClassName::class.java))
_INTENT_RE = re.compile(
    r"startActivity\s*\(\s*Intent\s*\(\s*\w+\s*,\s*(\w+)\s*::\s*class\.java\s*\)"
)
_CLICK_LISTENER_RE = re.compile(r"setOnClickListener")

# Android XML namespace URIs
_NS_ANDROID = "http://schemas.android.com/apk/res/android"
_NS_APP = "http://schemas.android.com/apk/res-auto"


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
        # When merging, preserve the most specific trigger: "button_tap" or
        # "deep_link" beat "programmatic" (XML nav graphs don't encode triggers).
        best: dict[tuple[str, str], GraphEdge] = {}
        for edge in raw:
            key = (edge.src_id, edge.dst_id)
            if key not in best:
                best[key] = edge
                continue
            existing = best[key]
            if edge.confidence >= existing.confidence:
                # New edge wins on confidence; inherit more specific trigger.
                trigger = edge.trigger
                if trigger == "programmatic" and existing.trigger != "programmatic":
                    trigger = existing.trigger
                best[key] = edge.model_copy(update={"trigger": trigger})
            elif existing.trigger == "programmatic" and edge.trigger != "programmatic":
                # Existing wins on confidence; upgrade its trigger.
                best[key] = existing.model_copy(update={"trigger": edge.trigger})
        return list(best.values())

    # ── Private helpers ───────────────────────────────────────────────────────

    def _from_code(
        self,
        source_files: list[SourceFile],
        symbols_by_file: dict[str, list[Symbol]],
        screen_by_name: dict[str, GraphNode],
    ) -> list[GraphEdge]:
        edges: list[GraphEdge] = []

        for sf in source_files:
            if sf.language != "kotlin":
                continue
            fp = str(sf.path)

            # Identify the source screen(s) declared in this file
            src_screens = [
                screen_by_name[s.name]
                for s in symbols_by_file.get(fp, [])
                if s.kind == "Class" and s.name in screen_by_name
            ]
            if not src_screens:
                continue
            src_screen = src_screens[0]

            try:
                text = sf.path.read_text(encoding="utf-8")
            except OSError:
                logger.warning("Cannot read %s", fp)
                continue

            for match in _INTENT_RE.finditer(text):
                dest_name = match.group(1)
                dst_screen = screen_by_name.get(dest_name)
                if dst_screen is None or dst_screen.id == src_screen.id:
                    continue

                # Infer trigger from a 300-char window before the match
                ctx_start = max(0, match.start() - 300)
                context = text[ctx_start : match.end()]
                trigger = "button_tap" if _CLICK_LISTENER_RE.search(context) else "programmatic"
                line = text[: match.start()].count("\n") + 1

                edges.append(
                    GraphEdge(
                        src_id=src_screen.id,
                        dst_id=dst_screen.id,
                        type=RelationType.NAVIGATES_TO,
                        confidence=0.8,
                        file_path=fp,
                        line=line,
                        trigger=trigger,
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
