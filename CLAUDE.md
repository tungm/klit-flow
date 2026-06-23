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
klit-flow serve [--port N] [--host H] # start MCP server (stdio) + web portal (default 127.0.0.1:5173; use --host 0.0.0.0 in Docker)
klit-flow status           # index freshness
klit-flow clean            # remove .klit-flow/ for the repo
```

### Docker

A self-contained image is provided (`Dockerfile`, `docker-compose.yml`). Parsers and the embedding model are baked in at build time, so the container makes **no network calls at runtime**.

```bash
# Build
docker build -t klit-flow .

# Index a repo mounted at /workspace
docker run --rm -v "$(pwd)/my-app:/workspace" klit-flow analyze /workspace --platform android

# Serve the portal (must bind 0.0.0.0 to be reachable from the host)
docker run --rm -p 5173:5173 -v "$(pwd)/my-app:/workspace" klit-flow serve --host 0.0.0.0
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

All `file_path` values are stored **relative to the project root** (the directory containing `.klit-flow`), as POSIX paths. `analyze` relativizes node/edge paths (`_relativize_nodes`/`_relativize_edges` in `cli.py`) just before persist/index/emit, so the whole project — graph DB, search index, emitted docs, and named-flow exports — can be moved without breaking the data. The walker reads from absolute paths; only the persisted/displayed paths are relative.

**Named flows** (`named_flows.py`) are user-authored, possibly **branching** screen flows created/edited in the web portal. A flow holds one or more *branches* (each an ordered path; branches typically share a prefix). They are persisted to `.klit-flow/named_flows.json` (a plain JSON file, **not** the graph DB) so they survive `analyze --force`, which deletes and rebuilds the graph DB. Legacy single-path records (a flat `screens` list) are migrated to a one-branch flow on load. The module is framework-agnostic; the web surface validates that consecutive screens within each branch are connected by real `NAVIGATES_TO` edges before saving, and search matches a queried screen sequence as an ordered subsequence (gaps allowed, case-insensitive) of **any one branch**.

## Architecture rules

- **Typed everything**: pydantic v2 models for config, nodes, and edges. No bare dicts across module boundaries.
- **`GraphStore` is the only persistence boundary**: never call LadybugDB directly outside `store.py`.
- **Graph schema** is authoritative in `graph/model.py`: node kinds (`File`, `Module`, `Function`, `Class`, `Method`, `Interface`, `Screen`); edge kinds (`DECLARES`, `IMPORTS`, `CALLS`, `EXTENDS`, `IMPLEMENTS`, `NAVIGATES_TO`).

## Release packaging rules

Every release **must** be fully usable on an air-gapped machine (no internet). This is non-negotiable.

### What must be bundled in every release

| Artifact | Location | Notes |
|----------|----------|-------|
| Embedding model | `release/vX.Y.Z/models/bge-small-en-v1.5/` | Platform-independent; use `klit-flow download-model` |
| Parser binaries | `release/vX.Y.Z/parsers/<platform>/` | One subdirectory per platform (see below) |
| Wheel | `release/vX.Y.Z/klit_flow-X.Y.Z-py3-none-any.whl` | Built with `python -m build` |
| Sdist | `release/vX.Y.Z/klit_flow-X.Y.Z.tar.gz` | Built with `python -m build` |

### Supported platforms (all must be present in parsers/)

`macos-arm64`, `macos-x86_64`, `linux-x86_64`, `linux-aarch64`, `windows-x86_64`, `windows-aarch64`

### How to build a release

```bash
# 0. Bump version in pyproject.toml and src/klit_flow/__init__.py
# 1. Install release extras
pip install -e ".[release]"

# 2. Download parsers for all platforms
klit-flow download-parsers --all-platforms --cache-dir release/vX.Y.Z/parsers

# 3. Download the embedding model
klit-flow download-model release/vX.Y.Z/models/bge-small-en-v1.5

# 4. Build wheel + sdist
python -m build --outdir release/vX.Y.Z

# 5. Quality gate must pass before publishing
ruff format --check . && ruff check . && pytest -q
```

### Runtime environment variables (users must set these on the target machine)

| Variable | Points to |
|----------|-----------|
| `KLIT_FLOW_PARSER_CACHE_DIR` | `parsers/` root — platform subdir is auto-detected at runtime |
| `KLIT_FLOW_MODEL_DIR` | `models/bge-small-en-v1.5/` directory |

### Key rules

- **Never** release without all 6 platform parser directories present.
- **Never** release without the `models/bge-small-en-v1.5/` directory present.
- Parser binaries are platform-specific compiled files (`.dylib`/`.so`/`.dll`) — do **not** assume binaries built on one OS work on another.
- The embedding model must **not** include `pytorch_model.bin` or `onnx/` — only `model.safetensors` + tokenizer files (prevents SafeTensor header errors).
- `KLIT_FLOW_MODEL_DIR` must use `local_files_only=True` in `SentenceTransformer` to prevent any HuggingFace network fallback.
- `KLIT_FLOW_PARSER_CACHE_DIR` auto-picks `<dir>/<platform>/` before falling back to `<dir>/` directly.
- Both env vars expand `~` via `Path.expanduser().resolve()` — never rely on the shell to expand them.

## Testing

- Fixtures live in `fixtures/mini_app/` — a tiny sample app for the target platform with 2–4 screens and known edges.
- Search/index tests must include an **offline assertion** (monkeypatch sockets to fail) to guarantee no network calls.
- CI: `ruff check && ruff format --check && pytest`.
