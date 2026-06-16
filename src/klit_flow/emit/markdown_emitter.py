"""Emit per-module and per-screen Markdown docs with YAML frontmatter.

Module docs  → ``out_dir/docs/modules/<filename>.md``
Screen docs  → ``out_dir/docs/screens/<ScreenName>.md``

Module frontmatter fields:
  id        — stable graph node ID
  kind      — always "Module"
  path      — source file path (POSIX)
  depends_on — list of File node IDs this module imports
  symbols   — list of Symbol node IDs declared in this module

Screen frontmatter fields:
  id            — stable graph node ID
  kind          — always "Screen"
  path          — source file path (POSIX)
  reachable_from — list of Screen node IDs that navigate to this screen
  navigates_to   — list of Screen node IDs this screen navigates to

Body: deterministic description.  When *summarizer* is supplied (Phase 9
``--summaries`` mode), a 1–2 sentence NL description is appended to the body
via a local Ollama model.  Off by default; behavior is unchanged when
``summarizer=None``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path, PurePosixPath

import frontmatter

from klit_flow.graph.model import GraphEdge, GraphNode, NodeKind, RelationType


def emit_module_docs(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    out_dir: Path,
    summarizer: Callable[[GraphNode], str | None] | None = None,
) -> list[Path]:
    """Write one ``.md`` per File node; return the list of written paths.

    Parameters
    ----------
    summarizer:
        Optional callable ``(GraphNode) -> str | None``.  When provided,
        its return value (if non-empty) is appended to the body as a NL
        summary paragraph.  Pass ``None`` (default) for deterministic output.
    """
    docs_dir = out_dir / "docs" / "modules"
    docs_dir.mkdir(parents=True, exist_ok=True)

    node_by_id: dict[str, GraphNode] = {n.id: n for n in nodes}

    # DECLARES edges: File → Symbol (direct symbol membership)
    file_symbols: dict[str, list[str]] = {}  # file_id → [symbol_id]
    # IMPORTS edges: File → File
    file_depends: dict[str, list[str]] = {}  # file_id → [dep_file_id]

    for edge in edges:
        if edge.type == RelationType.DECLARES:
            src = node_by_id.get(edge.src_id)
            if src and src.kind == NodeKind.File:
                file_symbols.setdefault(src.id, []).append(edge.dst_id)
        elif edge.type == RelationType.IMPORTS:
            file_depends.setdefault(edge.src_id, []).append(edge.dst_id)

    written: list[Path] = []

    for node in nodes:
        if node.kind != NodeKind.File:
            continue

        sym_ids = file_symbols.get(node.id, [])
        dep_ids = file_depends.get(node.id, [])

        # Build deterministic body
        sym_names = [node_by_id[sid].name for sid in sym_ids if sid in node_by_id]
        lines: list[str] = [f"## {node.name}", ""]
        if sym_names:
            lines.append("### Symbols")
            for name in sorted(sym_names):
                lines.append(f"- {name}")
        else:
            lines.append("_No symbols extracted._")

        # Optional NL summary (Phase 9 --summaries mode)
        if summarizer is not None:
            summary = summarizer(node)
            if summary:
                lines += ["", "### Summary", "", summary]

        post = frontmatter.Post(
            "\n".join(lines),
            id=node.id,
            kind="Module",
            path=PurePosixPath(node.file_path).as_posix(),
            depends_on=dep_ids,
            symbols=sym_ids,
        )

        slug = node.name.replace(".", "_")
        out_path = docs_dir / f"{slug}.md"
        out_path.write_text(frontmatter.dumps(post), encoding="utf-8")
        written.append(out_path)

    return written


def emit_screen_docs(
    screen_nodes: list[GraphNode],
    edges: list[GraphEdge],
    out_dir: Path,
    summarizer: Callable[[GraphNode], str | None] | None = None,
) -> list[Path]:
    """Write one ``.md`` per Screen node to ``out_dir/docs/screens/``.

    Frontmatter includes ``reachable_from`` and ``navigates_to`` Screen IDs.

    Parameters
    ----------
    summarizer:
        Optional callable ``(GraphNode) -> str | None``.  When provided,
        its return value (if non-empty) is appended to the body as a NL
        summary paragraph.
    """
    if not screen_nodes:
        return []

    docs_dir = out_dir / "docs" / "screens"
    docs_dir.mkdir(parents=True, exist_ok=True)

    screen_ids = {n.id for n in screen_nodes}

    navigates_to: dict[str, list[str]] = {}
    reachable_from: dict[str, list[str]] = {}
    for edge in edges:
        if edge.type == RelationType.NAVIGATES_TO:
            if edge.src_id in screen_ids and edge.dst_id in screen_ids:
                navigates_to.setdefault(edge.src_id, []).append(edge.dst_id)
                reachable_from.setdefault(edge.dst_id, []).append(edge.src_id)

    written: list[Path] = []
    for screen in screen_nodes:
        nav_to = navigates_to.get(screen.id, [])
        reach_from = reachable_from.get(screen.id, [])

        lines: list[str] = [f"## {screen.name}", ""]
        if nav_to:
            lines.append(f"Navigates to {len(nav_to)} screen(s).")
        if reach_from:
            lines.append(f"Reachable from {len(reach_from)} screen(s).")
        if not nav_to and not reach_from:
            lines.append("_No navigation edges detected._")

        # Optional NL summary (Phase 9 --summaries mode)
        if summarizer is not None:
            summary = summarizer(screen)
            if summary:
                lines += ["", "### Summary", "", summary]

        post = frontmatter.Post(
            "\n".join(lines),
            id=screen.id,
            kind="Screen",
            path=PurePosixPath(screen.file_path).as_posix(),
            reachable_from=reach_from,
            navigates_to=nav_to,
        )

        out_path = docs_dir / f"{screen.name}.md"
        out_path.write_text(frontmatter.dumps(post), encoding="utf-8")
        written.append(out_path)

    return written
