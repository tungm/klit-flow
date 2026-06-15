"""Emit per-module Markdown docs with YAML frontmatter.

Output: ``out_dir/docs/modules/<filename>.md``

Frontmatter fields:
  id        — stable graph node ID
  kind      — always "Module"
  path      — source file path (POSIX, relative to target root when possible)
  depends_on — list of File node IDs this module imports
  symbols   — list of Symbol node IDs declared in this module

Body: deterministic symbol list (no LLM; NL prose is Phase 9).
"""

from pathlib import Path, PurePosixPath

import frontmatter

from klit_flow.graph.model import GraphEdge, GraphNode, NodeKind, RelationType


def emit_module_docs(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    out_dir: Path,
) -> list[Path]:
    """Write one ``.md`` per File node; return the list of written paths."""
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
