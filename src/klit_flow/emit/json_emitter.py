"""Emit the full graph as graph.json."""

import json
from pathlib import Path

from klit_flow.graph.model import GraphEdge, GraphNode


def emit_graph_json(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    out_dir: Path,
) -> Path:
    """Write ``out_dir/graph.json`` with all nodes and edges.

    Returns the path of the written file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "graph.json"
    data = {
        "nodes": [n.model_dump() for n in nodes],
        "edges": [e.model_dump() for e in edges],
    }
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
