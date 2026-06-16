# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`klit-flow` is a local, offline code-intelligence tool that extracts dependency graphs and screen-navigation (user flow) graphs from a mobile app's source code, emitting both human-readable and AI-indexable artifacts. Python 3.11+. MIT license.

## Non-negotiable constraints

- The `analyze` pipeline is **deterministic**: no LLM calls, no network. Ever.
- Embeddings run **locally** via `sentence-transformers`. No cloud APIs.
- Optional NL summaries use a **local Ollama model only**, behind `--summaries`. Off by default.
- No telemetry, no secrets, no outbound calls anywhere in the codebase.
- `IMPORTS` edges must be exact. `CALLS` and `NAVIGATES_TO` are best-effort and must carry an honest `confidence` score; never fabricate edges.
- Parse failures are logged and skipped — never fatal to the run.

## Key commands

```bash
# Install (dev)
pip install -e ".[dev]"

# Quality gate (run after every change)
ruff format --check . && ruff check . && pytest -q

# Or via the slash command:
/check

# Index a target repo
klit-flow analyze <path> --platform <android|ios|react_native|flutter>

# Other CLI commands
klit-flow query "<text>"   # hybrid search
klit-flow flows [<screen>] # list NAVIGATES_TO edges; filter by screen name
klit-flow serve            # start MCP server over stdio
klit-flow status           # index freshness
klit-flow clean            # remove .klit-flow/ for the repo
```

## How we work

- Build strictly **phase by phase** per `PLAN.md`. Do not scaffold future phases.
- After each phase: extend tests, run `/check` (ruff + pytest), keep green.
- Stop for review at the end of each phase.
- See `PLAN.md` for the full spec and current phase definitions.

## Architecture

The indexing pipeline runs in this order:

```
walk → parse → resolve → (flows) → persist → index → emit
```

1. **Walker** (`walker.py`) — discovers source files, honors `.gitignore` + `.klit-flowignore`.
2. **Parser** (`parsing/`) — tree-sitter per file via `tree-sitter-language-pack`; `.scm` query files per language; emits `Symbol` records.
3. **Resolver** (`graph/resolver.py`) — turns imports + call sites into typed graph edges.
4. **Flow extractor** (`flows/`) — platform-specific ABC; adds `Screen` nodes and `NAVIGATES_TO` edges; also parses declarative nav files (XML, storyboard).
5. **GraphStore** (`graph/store.py`) — ABC interface; current impl is `LadybugGraphStore` (LadybugDB, MIT fork of archived Kuzu v0.11.3; `import ladybug`). All persistence goes through this interface so the engine is swappable.
6. **Index** (`index/`) — local `sentence-transformers` embeddings (default `BAAI/bge-small-en-v1.5`) + BM25 (`rank-bm25`); hybrid RRF fusion in `search.py`.
7. **Emitters** (`emit/`) — `graph.json`, per-module/screen Markdown with YAML frontmatter, Mermaid diagrams (`dependencies.mmd`, `flows.mmd`).
8. **MCP server** (`server/mcp_server.py`) — FastMCP tools: `query`, `context`, `impact`, `flows`, `cypher`.

Index artifacts are written to `.klit-flow/` inside the target repo (gitignored). Output docs go to `.klit-flow/out/`.

## Architecture rules

- **Typed everything**: pydantic v2 models for config, nodes, and edges. No bare dicts across module boundaries.
- **`GraphStore` is the only persistence boundary**: never call LadybugDB directly outside `store.py`.
- **Graph schema** is authoritative in `graph/model.py`: node kinds (`File`, `Module`, `Function`, `Class`, `Method`, `Interface`, `Screen`); edge kinds (`DECLARES`, `IMPORTS`, `CALLS`, `EXTENDS`, `IMPLEMENTS`, `NAVIGATES_TO`).

## Testing

- Fixtures live in `fixtures/mini_app/` — a tiny sample app for the target platform with 2–4 screens and known edges.
- Search/index tests must include an **offline assertion** (monkeypatch sockets to fail) to guarantee no network calls.
- CI: `ruff check && ruff format --check && pytest`.
