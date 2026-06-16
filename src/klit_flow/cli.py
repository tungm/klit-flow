import typer

app = typer.Typer(
    name="klit-flow",
    help="Local, offline code intelligence for mobile apps.",
    no_args_is_help=True,
)


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
    typer.echo("analyze: not yet implemented")


@app.command()
def query(
    text: str = typer.Argument(..., help="Natural-language search query."),
) -> None:
    """Hybrid search (semantic + BM25) against the index."""
    typer.echo("query: not yet implemented")


@app.command()
def serve() -> None:
    """Start the MCP server over stdio."""
    from klit_flow.server.mcp_server import main

    main()


@app.command()
def status() -> None:
    """Show index freshness for the target repo."""
    typer.echo("status: not yet implemented")


@app.command()
def clean() -> None:
    """Remove the .klit-flow/ index directory for the target repo."""
    typer.echo("clean: not yet implemented")


if __name__ == "__main__":
    app()
