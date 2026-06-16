from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import typer

app = typer.Typer(
    name="klit-flow",
    help="Local, offline code intelligence for mobile apps.",
    no_args_is_help=True,
)

_KLIT_DIR = ".klit-flow"
_DB_NAME = "graph.db"
_BM25_NAME = "bm25.pkl"
_OUT_DIR = "out"


def _klit_paths(target: Path) -> tuple[Path, Path, Path, Path]:
    """Return (klit_dir, db_path, bm25_path, out_dir) for a target repo root."""
    klit_dir = target / _KLIT_DIR
    return klit_dir, klit_dir / _DB_NAME, klit_dir / _BM25_NAME, klit_dir / _OUT_DIR


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


@app.command()
def analyze(
    path: str = typer.Argument(..., help="Path to the target repository."),
    platform: str = typer.Option(
        ..., "--platform", "-p", help="Target platform: android|ios|react_native|flutter"
    ),
    summaries: bool = typer.Option(
        False, "--summaries", help="Generate local NL summaries via Ollama."
    ),
    force: bool = typer.Option(False, "--force", help="Re-index even if the index is fresh."),
) -> None:
    """Index a mobile app source tree and emit dependency + flow graphs."""
    from klit_flow.emit.json_emitter import emit_graph_json
    from klit_flow.emit.markdown_emitter import emit_module_docs, emit_screen_docs
    from klit_flow.emit.mermaid_emitter import emit_dependency_diagram, emit_flow_diagram
    from klit_flow.flows import get_extractor
    from klit_flow.graph.resolver import resolve
    from klit_flow.graph.store import LadybugGraphStore
    from klit_flow.index.embeddings import Embedder
    from klit_flow.index.search import build_index
    from klit_flow.parsing.extractor import extract
    from klit_flow.summarize import Summarizer
    from klit_flow.walker import walk

    root = Path(path).resolve()
    if not root.exists():
        typer.echo(f"Error: path not found: {root}", err=True)
        raise typer.Exit(1)

    klit_dir, db_path, bm25_path, out_dir = _klit_paths(root)

    if db_path.exists() and not force:
        typer.echo("Already indexed. Use --force to re-index.")
        raise typer.Exit(0)

    if db_path.exists() and force:
        shutil.rmtree(db_path)

    klit_dir.mkdir(parents=True, exist_ok=True)

    # ── Walk ──────────────────────────────────────────────────────────────────
    typer.echo(f"Walking {root} …")
    source_files = walk(root)
    typer.echo(f"  {len(source_files)} source files found.")

    # ── Parse ─────────────────────────────────────────────────────────────────
    typer.echo("Parsing symbols …")
    symbols_by_file = {str(sf.path): extract(sf.path, sf.language) for sf in source_files}

    # ── Resolve ───────────────────────────────────────────────────────────────
    typer.echo("Resolving graph edges …")
    nodes, edges = resolve(source_files, symbols_by_file)

    # ── Flows ─────────────────────────────────────────────────────────────────
    typer.echo(f"Extracting {platform} screen flows …")
    extractor = get_extractor(platform)
    screen_nodes = extractor.extract_screens(source_files, symbols_by_file)
    flow_edges = extractor.extract_flows(source_files, symbols_by_file, screen_nodes)
    nodes = nodes + screen_nodes
    edges = edges + flow_edges
    typer.echo(f"  {len(screen_nodes)} screens, {len(flow_edges)} navigation edges.")

    # ── Persist ───────────────────────────────────────────────────────────────
    typer.echo("Persisting graph …")
    with LadybugGraphStore(db_path) as store:
        store.create_schema()
        store.add_nodes(nodes)
        store.add_edges(edges)

        # ── Index ─────────────────────────────────────────────────────────────
        typer.echo("Building search index …")
        embedder = Embedder()
        bm25 = build_index(nodes, store, embedder)
        bm25.save(bm25_path)

    # ── Emit ──────────────────────────────────────────────────────────────────
    typer.echo("Emitting artifacts …")
    summarizer = Summarizer() if summaries else None
    emit_graph_json(nodes, edges, out_dir)
    emit_module_docs(nodes, edges, out_dir, summarizer=summarizer)
    emit_screen_docs(screen_nodes, flow_edges, out_dir, summarizer=summarizer)
    emit_dependency_diagram(nodes, edges, out_dir)
    emit_flow_diagram(screen_nodes, flow_edges, out_dir)

    typer.echo(f"\nDone. {len(nodes)} nodes, {len(edges)} edges → {out_dir.relative_to(root)}")


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


@app.command()
def query(
    text: str = typer.Argument(..., help="Natural-language search query."),
    k: int = typer.Option(10, "--top", "-n", help="Number of results to return."),
) -> None:
    """Hybrid search (semantic + BM25) against the index."""
    from klit_flow.graph.store import LadybugGraphStore
    from klit_flow.index.bm25 import BM25Index
    from klit_flow.index.embeddings import Embedder
    from klit_flow.index.search import hybrid_search

    _, db_path, bm25_path, _ = _klit_paths(Path.cwd())

    if not db_path.exists():
        typer.echo("No index found. Run 'klit-flow analyze' first.", err=True)
        raise typer.Exit(1)

    with LadybugGraphStore(db_path) as store:
        bm25 = BM25Index.load(bm25_path) if bm25_path.exists() else _empty_bm25()
        embedder = Embedder()
        node_ids = hybrid_search(text, bm25, store, embedder, k=k)

        if not node_ids:
            typer.echo("No results.")
            return

        typer.echo(f"{'KIND':<12} {'NAME':<40} FILE")
        typer.echo("-" * 80)
        for nid in node_ids:
            rows = store.query(
                f"MATCH (n:KlitNode {{id: '{nid}'}}) RETURN n.kind, n.name, n.file_path"
            )
            if rows:
                kind, name, fp = rows[0]
                typer.echo(f"{kind:<12} {name:<40} {fp}")


# ---------------------------------------------------------------------------
# flows
# ---------------------------------------------------------------------------


@app.command()
def flows(
    screen: str = typer.Argument("", help="Screen name to filter by (leave empty for all edges)."),
) -> None:
    """List NAVIGATES_TO edges. Optionally filter to edges involving a screen."""
    from klit_flow.graph.store import LadybugGraphStore

    _, db_path, _, _ = _klit_paths(Path.cwd())

    if not db_path.exists():
        typer.echo("No index found. Run 'klit-flow analyze' first.", err=True)
        raise typer.Exit(1)

    with LadybugGraphStore(db_path) as store:
        if screen:
            s = screen.replace("'", "\\'")
            rows = store.query(
                f"MATCH (a:KlitNode)-[e:KlitEdge]->(b:KlitNode) "
                f"WHERE e.type = 'NAVIGATES_TO' AND (a.name = '{s}' OR b.name = '{s}') "
                f"RETURN a.name, b.name, e.trigger, e.confidence "
                f"ORDER BY a.name, b.name"
            )
        else:
            rows = store.query(
                "MATCH (a:KlitNode)-[e:KlitEdge]->(b:KlitNode) "
                "WHERE e.type = 'NAVIGATES_TO' "
                "RETURN a.name, b.name, e.trigger, e.confidence "
                "ORDER BY a.name, b.name"
            )

    if not rows:
        msg = (
            f"No navigation edges found for '{screen}'." if screen else "No navigation edges found."
        )
        typer.echo(msg)
        return

    typer.echo(f"{'FROM':<30} {'TO':<30} {'TRIGGER':<16} CONF")
    typer.echo("-" * 82)
    for src_name, dst_name, trigger, confidence in rows:
        typer.echo(f"{src_name:<30} {dst_name:<30} {trigger:<16} {confidence:.2f}")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@app.command()
def serve() -> None:
    """Start the MCP server over stdio."""
    from klit_flow.server.mcp_server import main

    main()


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@app.command()
def status() -> None:
    """Show index freshness for the target repo."""
    from klit_flow.graph.store import LadybugGraphStore

    _, db_path, bm25_path, out_dir = _klit_paths(Path.cwd())

    if not db_path.exists():
        typer.echo("No index found. Run 'klit-flow analyze' first.")
        raise typer.Exit(1)

    mtime = datetime.fromtimestamp(db_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    with LadybugGraphStore(db_path) as store:
        node_count = store.query("MATCH (n:KlitNode) RETURN count(n)")[0][0]
        edge_count = store.query("MATCH ()-[e:KlitEdge]->() RETURN count(e)")[0][0]
        screen_count = store.query("MATCH (n:KlitNode) WHERE n.kind = 'Screen' RETURN count(n)")[0][
            0
        ]

    typer.echo(f"Index:    {db_path}")
    typer.echo(f"Updated:  {mtime}")
    typer.echo(f"Nodes:    {node_count}")
    typer.echo(f"Edges:    {edge_count}")
    typer.echo(f"Screens:  {screen_count}")
    typer.echo(f"BM25:     {'present' if bm25_path.exists() else 'missing'}")
    typer.echo(f"Artifacts:{out_dir}")


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------


@app.command()
def clean() -> None:
    """Remove the .klit-flow/ index directory for the target repo."""
    klit_dir, _, _, _ = _klit_paths(Path.cwd())

    if not klit_dir.exists():
        typer.echo("Nothing to clean.")
        raise typer.Exit(0)

    shutil.rmtree(klit_dir)
    typer.echo(f"Removed {klit_dir}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_bm25():
    from klit_flow.index.bm25 import BM25Index

    idx = BM25Index()
    idx.build()
    return idx


if __name__ == "__main__":
    app()
