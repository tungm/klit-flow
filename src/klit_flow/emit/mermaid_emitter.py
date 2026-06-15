"""Emit Mermaid flowchart diagrams.

Phase 4: ``diagrams/dependencies.mmd`` — module-level import graph.
Phase 5: ``diagrams/flows.mmd``        — screen navigation graph.
"""

from pathlib import Path

from klit_flow.graph.model import GraphEdge, GraphNode, NodeKind, RelationType


def _mmd_id(node_id: str) -> str:
    """Convert a graph node ID to a valid Mermaid node identifier."""
    return f"n{node_id}"


def _mmd_label(name: str) -> str:
    return name.replace('"', "'")


def emit_dependency_diagram(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    out_dir: Path,
) -> Path:
    """Write ``out_dir/diagrams/dependencies.mmd``.

    Produces a ``flowchart LR`` of File→File IMPORTS edges.
    Returns the path of the written file.
    """
    diag_dir = out_dir / "diagrams"
    diag_dir.mkdir(parents=True, exist_ok=True)
    out = diag_dir / "dependencies.mmd"

    file_nodes: dict[str, GraphNode] = {n.id: n for n in nodes if n.kind == NodeKind.File}

    import_edges = [
        e
        for e in edges
        if e.type == RelationType.IMPORTS and e.src_id in file_nodes and e.dst_id in file_nodes
    ]

    lines: list[str] = ["flowchart LR"]

    # Declare nodes that appear in at least one import edge
    seen: set[str] = set()
    for e in import_edges:
        for nid in (e.src_id, e.dst_id):
            if nid not in seen:
                label = _mmd_label(file_nodes[nid].name)
                lines.append(f'  {_mmd_id(nid)}["{label}"]')
                seen.add(nid)

    # Edges
    for e in import_edges:
        lines.append(f"  {_mmd_id(e.src_id)} --> {_mmd_id(e.dst_id)}")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def emit_flow_diagram(
    screen_nodes: list[GraphNode],
    edges: list[GraphEdge],
    out_dir: Path,
) -> Path:
    """Write ``out_dir/diagrams/flows.mmd``.

    Produces a ``flowchart LR`` of Screen→Screen NAVIGATES_TO edges,
    with edge labels showing the navigation trigger.
    """
    diag_dir = out_dir / "diagrams"
    diag_dir.mkdir(parents=True, exist_ok=True)
    out = diag_dir / "flows.mmd"

    screen_by_id: dict[str, GraphNode] = {n.id: n for n in screen_nodes}

    nav_edges = [
        e
        for e in edges
        if e.type == RelationType.NAVIGATES_TO
        and e.src_id in screen_by_id
        and e.dst_id in screen_by_id
    ]

    lines: list[str] = ["flowchart LR"]

    seen: set[str] = set()
    for e in nav_edges:
        for nid in (e.src_id, e.dst_id):
            if nid not in seen:
                label = _mmd_label(screen_by_id[nid].name)
                lines.append(f'  {_mmd_id(nid)}["{label}"]')
                seen.add(nid)

    for e in nav_edges:
        trigger = _mmd_label(e.trigger or "")
        if trigger:
            lines.append(f'  {_mmd_id(e.src_id)} -->|"{trigger}"| {_mmd_id(e.dst_id)}')
        else:
            lines.append(f"  {_mmd_id(e.src_id)} --> {_mmd_id(e.dst_id)}")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
