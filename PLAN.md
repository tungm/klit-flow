# Implementation Plan: `klit-flow` — Local Code Intelligence for Mobile Apps

> A private, offline tool that extracts **dependencies** and **user (screen) flows**
> from an undocumented mobile app's source code, and emits both **human-readable**
> and **AI-indexable** artifacts. Inspired by GitNexus, rebuilt from scratch in
> Python so it is free of any third-party license restrictions.

---

## How to use this file with Claude Code

This is a phased build spec. Work through it **one phase at a time, in order**.
For each phase:

1. Read the **Tasks** and **Acceptance criteria**.
2. Implement only that phase. Do not scaffold future phases ahead of time.
3. Write/extend the tests listed for the phase and make them pass before moving on.
4. Run `ruff check`, `ruff format`, and `pytest` after each phase; keep the tree green.
5. Pause after each phase so the changes can be reviewed.

**Before starting Phase 5, set the target platform** (see "Configuration"). The
mobile flow extractor is platform-specific; only the targeted module needs to work.

Conventions and guardrails are at the bottom — follow them throughout.

---

## 0. Project license

klit-flow is released under the **MIT License**.

```
MIT License

Copyright (c) 2026 klit-flow contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Phase 0 task**: write this text verbatim to `LICENSE` at the repo root.

**Why MIT**: all runtime dependencies (tree-sitter, LadybugDB, networkx,
sentence-transformers, pydantic, typer, rank-bm25, python-frontmatter, mcp,
python-louvain, ollama) are MIT, Apache 2.0, or BSD — all permissive and
compatible with MIT. No copyleft (GPL/LGPL) dependency is included; `leidenalg`
was explicitly excluded for this reason and replaced with `python-louvain` (BSD).
The result is a stack with **no license obligations on the consuming enterprise**
beyond attribution in the `LICENSE` file.

**Attribution requirement (the only obligation)**: the MIT license requires that
the `LICENSE` file (with its copyright notice) is included in all copies or
substantial portions of the software. This is satisfied automatically by
committing the `LICENSE` file to the repository.

---

## 1. Goals and non-goals

### Goals
- Parse a mobile app source tree (no documentation required) and produce a
  **knowledge graph** of symbols and their relationships.
- Extract two layers:
  - **Dependency layer** — import edges and call edges between files/symbols.
  - **Flow layer** — screen nodes and screen-to-screen navigation edges (the user-flow graph).
- Emit outputs that are simultaneously:
  - **Human-readable**: Markdown with YAML frontmatter + Mermaid diagrams.
  - **Machine/AI-queryable**: a JSON graph + an embedded graph DB + a local semantic index.
- Expose the graph to AI agents via an **MCP server** so it can be queried later.

### Non-goals (explicitly out of scope for v1)
- Dynamic analysis / runtime tracing. Static analysis only.
- 100% call-resolution accuracy. Import-level edges must be exact; call-level edges
  are best-effort with confidence scores.
- Cloud LLM usage during indexing. See constraints.
- A graphical UI. CLI + MCP only.

---

## 2. Hard constraints

- **Privacy**: everything runs locally. No source code, file contents, or symbol
  names may leave the machine. No outbound network calls in the indexing path.
- **Low AI quota**: the indexing pipeline must be **fully deterministic — zero LLM
  calls**. Embeddings are computed with a **local** model (sentence-transformers).
  Any optional natural-language summarization must use a **local** model via Ollama,
  never a cloud API.
- **Self-contained**: a single `pip install` (plus an optional local model download)
  should make the tool runnable offline.

---

## 3. Tech stack

Target **Python 3.11+**.

| Concern | Library | Notes |
|---|---|---|
| Parsing | `tree-sitter` + `tree-sitter-language-pack` | One dependency provides grammars for Kotlin, Swift, Dart, TS/JS, Java, etc. |
| Graph store | `ladybug` (LadybugDB) | Embedded graph DB, Cypher + vector index, no server. Maintained MIT fork of Kuzu, which was archived Oct 2025; API is equivalent to Kuzu v0.11.3, so usage is `import ladybug` in place of `import kuzu`. Access only through the `GraphStore` interface (Phase 6) so the engine stays swappable. |
| In-memory graph | `networkx` | Build/traverse before persisting; also used for clustering. |
| Embeddings (local) | `sentence-transformers` | Default model `BAAI/bge-small-en-v1.5` (small, CPU-friendly). Swap to a code model later if desired. |
| Lexical search | `rank-bm25` | BM25 over symbol names + docstrings. |
| Config & models | `pydantic` v2 | Typed config and graph node/edge models. |
| CLI | `typer` | Subcommands: `analyze`, `query`, `serve`, `clean`, `status`. |
| Markdown frontmatter | `python-frontmatter` | Emit/parse `.md` with YAML frontmatter. |
| MCP server | `mcp` (official SDK, FastMCP) | Exposes query tools over stdio. |
| Clustering (optional) | `python-louvain` | Functional grouping of symbols (BSD-licensed). `leidenalg` was the original candidate but is GPL v3 — excluded to keep the project commercially usable. |
| Local summaries (optional) | `ollama` python client | Only if NL summaries are enabled; off by default. |
| Lint/format/test | `ruff`, `pytest` | Keep green after every phase. |

Pin exact versions in `pyproject.toml` at scaffold time and commit the lock.

---

## 4. Architecture (indexing pipeline)

Deterministic phases, mirroring a GitNexus-style pipeline:

```
walk → parse → resolve → (flows) → persist → index(embeddings+bm25) → emit
```

1. **Walk** — discover source files, honor `.gitignore` + a `.klit-flowignore`.
2. **Parse** — tree-sitter per file → raw symbols (functions, classes, methods, imports).
3. **Resolve** — turn imports + call sites into graph edges across files.
4. **Flows** — platform-specific extractor adds Screen nodes + NAVIGATES_TO edges.
5. **Persist** — write the graph via the `GraphStore` interface to LadybugDB (and `graph.json`).
6. **Index** — compute local embeddings + BM25 indexes for hybrid search.
7. **Emit** — write Markdown (frontmatter) + Mermaid diagrams.

Index artifacts live in `.klit-flow/` inside the target repo (gitignored).

---

## 5. Repository structure

```
klit-flow/
  pyproject.toml
  README.md
  LICENSE                    # MIT License (see Section 0)
  CLAUDE.md                  # persistent context + guardrails (auto-loaded by Claude Code)
  .claude/
    settings.json            # permissions, env, model defaults (committed)
    commands/
      check.md               # /check -> run ruff + pytest
  src/klit_flow/
    __init__.py
    cli.py                   # typer entrypoint
    config.py                # pydantic Settings; target platform, paths, model name
    walker.py                # file discovery + ignore handling
    parsing/
      __init__.py
      registry.py            # extension -> language -> grammar
      extractor.py           # run tree-sitter queries -> Symbol records
      queries/               # .scm tree-sitter query files, one per language
    graph/
      __init__.py
      model.py               # pydantic Node/Edge types + enums (schema)
      resolver.py            # import + call resolution -> edges
      store.py               # GraphStore interface + LadybugDB impl + Cypher helpers
    flows/
      __init__.py
      base.py                # ScreenFlowExtractor ABC
      android.py             # Kotlin/Android
      ios.py                 # Swift/iOS
      react_native.py        # RN (TS/JS)
      flutter.py             # Dart/Flutter
    index/
      __init__.py
      embeddings.py          # local sentence-transformers
      bm25.py
      search.py              # RRF fusion of BM25 + semantic
    emit/
      __init__.py
      json_emitter.py
      markdown_emitter.py
      mermaid_emitter.py
    server/
      __init__.py
      mcp_server.py          # FastMCP tools
  tests/
    test_walker.py
    test_extractor_<lang>.py
    test_resolver.py
    test_flows_<platform>.py
    test_emit.py
    test_search.py
  fixtures/
    mini_app/                # tiny sample app used by tests (target platform)
```

---

## 6. Graph data model

Define in `graph/model.py` with pydantic + enums.

**Node kinds**: `File`, `Module`, `Function`, `Class`, `Method`, `Interface`, `Screen`.

Common node fields: `id` (stable hash of kind+path+name), `kind`, `name`,
`file_path`, `start_line`, `end_line`, `language`.

**Edge kinds (`RelationType`)**:
- `DECLARES` — File/Class declares a Symbol.
- `IMPORTS` — File imports Module/Symbol. **Must be exact.**
- `CALLS` — Symbol calls Symbol. Best-effort; carries `confidence` (0–1).
- `EXTENDS` / `IMPLEMENTS` — inheritance / interface.
- `NAVIGATES_TO` — Screen → Screen. Carries `trigger` (e.g. `button_tap`,
  `deep_link`, `programmatic`) and `confidence`.

Common edge fields: `src_id`, `dst_id`, `type`, `confidence`, `file_path`,
`line`, plus type-specific properties (`trigger`).

Keep this schema authoritative; the LadybugDB DDL and JSON emitter derive from it.

---

## 7. Output format spec

Written to `.klit-flow/out/`:

- `graph.json` — `{ "nodes": [...], "edges": [...] }`, the full graph. Primary
  machine-readable artifact for structural queries.
- `docs/modules/<module>.md` — one file per module/file with YAML frontmatter:
  ```yaml
  ---
  id: <node id>
  kind: Module
  path: src/feature/auth.kt
  depends_on: [<module ids>]
  symbols: [<symbol ids>]
  ---
  ```
  Body: a short deterministic description (symbol list, inbound/outbound edges).
  NL prose only if local-summary mode is enabled.
- `docs/screens/<screen>.md` — one file per Screen with `reachable_from` /
  `navigates_to` in frontmatter.
- `diagrams/dependencies.mmd` — Mermaid `flowchart` of the module dependency graph.
- `diagrams/flows.mmd` — Mermaid `flowchart` of the screen navigation graph.

Frontmatter is what makes these dual-purpose: humans read the body; RAG indexes
chunk cleanly on the structured fields.

---

## 8. Mobile flow extractor (the differentiator)

GitNexus traces code-level call flows, not screen navigation. This module is the
project's main value-add. Design it as pluggable: a `ScreenFlowExtractor` ABC with
one implementation per platform, selected by `config.platform`.

Each extractor does two things: identify **Screen nodes** and emit **NAVIGATES_TO**
edges (with a `trigger` where detectable). Some navigation is declarative (Android
nav-graph XML, iOS storyboard segues) — parse those files directly **in addition to**
tree-sitter on code.

Pattern reference per platform:

### Android (Kotlin)
- **Screens**: classes extending `Activity` / `Fragment`; `@Composable` functions
  used as navigation destinations.
- **Nav edges**: `startActivity(Intent(this, X::class.java))`;
  `navController.navigate(R.id.x)` or `navigate("route")`; **also parse**
  `res/navigation/*.xml` (`<fragment>`, `<action app:destination=...>`).
- **Triggers**: `setOnClickListener {}`, Compose `Button(onClick = ...)`.

### iOS (Swift)
- **Screens**: `UIViewController` subclasses; SwiftUI `View` structs.
- **Nav edges**: `pushViewController`, `present(_:animated:)`, `performSegue`;
  SwiftUI `NavigationLink`, `.navigationDestination`, `.sheet`, `.fullScreenCover`;
  **also parse** `*.storyboard` segues (XML).
- **Triggers**: `@IBAction`, SwiftUI `Button { } action`.

### React Native (TS/JS)
- **Screens**: components registered as `<Stack.Screen name="X" .../>`.
- **Nav edges**: `navigation.navigate('X')`, `navigation.push('X')`,
  `navigation.replace('X')`; navigator config blocks.
- **Triggers**: `onPress`.

### Flutter (Dart)
- **Screens**: `StatelessWidget` / `StatefulWidget` subclasses used as routes.
- **Nav edges**: `Navigator.push(MaterialPageRoute(builder: ...))`,
  `Navigator.pushNamed('/x')`; named-routes table in `MaterialApp(routes: {...})`;
  GoRouter / AutoRoute route definitions.
- **Triggers**: `onPressed`, `onTap`, `GestureDetector`.

**Known limitations to record in output** (set `confidence < 1` and note in docs):
deep links, conditional/dynamic routing, and string-built route names cannot be
fully resolved statically.

---

## 9. Phased roadmap

### Phase 0 — Scaffold
**Tasks**: create the repo structure, `pyproject.toml` with pinned deps, `ruff` +
`pytest` config, `cli.py` with empty `analyze/query/serve/status/clean` commands,
`config.py` with a `Settings` model (target repo path, `platform`, model name,
output dir).

Also scaffold the Claude Code config (keep it minimal — CLAUDE.md + settings.json
are all that's needed; do not add hooks/subagents):

- **`CLAUDE.md`** at the repo root with this content (auto-loaded every session,
  survives `/compact`, so the guardrails persist):

  ```markdown
  # klit-flow — project memory

  Local, offline code-intelligence tool: extracts dependency and screen-flow
  graphs from a mobile app's source and emits human- + AI-readable artifacts.

  ## Non-negotiable constraints
  - The `analyze` pipeline is DETERMINISTIC: no LLM calls, no network. Ever.
  - Embeddings run LOCALLY via sentence-transformers. No cloud APIs.
  - Optional NL summaries use a LOCAL Ollama model only, behind `--summaries`.
  - No telemetry, no secrets, no outbound calls anywhere in the codebase.

  ## How we work
  - Build strictly phase by phase per PLAN.md. Do not scaffold future phases early.
  - After each phase: extend tests, then run `/check` (ruff + pytest). Keep green.
  - Stop for review at the end of each phase.

  ## Architecture rules
  - Typed everything (pydantic models for config, nodes, edges). No bare dicts
    across module boundaries.
  - `IMPORTS` edges must be exact. `CALLS`/`NAVIGATES_TO` are best-effort and MUST
    carry an honest `confidence`; never fabricate edges.
  - Parse failures are logged and skipped, never fatal.
  - All graph persistence goes through the `GraphStore` interface (graph engine
    is swappable; current impl = LadybugDB).

  ## Key commands
  - Index: `klit-flow analyze <path> --platform <android|ios|react_native|flutter>`
  - Quality gate: `/check`

  See @PLAN.md for the full spec and phase definitions.
  ```

- **`.claude/settings.json`** — a minimal, committed config granting the quality-gate
  commands and nothing networked. Confirm the exact permission syntax against the
  current docs (https://code.claude.com/docs/en/claude-directory) at scaffold time:

  ```json
  {
    "permissions": {
      "allow": ["Bash(pytest:*)", "Bash(ruff:*)", "Bash(python:*)"]
    }
  }
  ```

- **`.claude/commands/check.md`** — a `/check` slash command:

  ```markdown
  Run the full quality gate and report failures concisely:
  `ruff format --check . && ruff check . && pytest -q`
  ```

**Acceptance**: `klit-flow --help` lists all subcommands; `pytest` runs (zero tests
OK); `CLAUDE.md` and `.claude/settings.json` exist and are committed; `/check` runs
the gate.

### Phase 1 — Walker
**Tasks**: implement `walker.py` — recursive discovery, honor `.gitignore` and
`.klit-flowignore`, language detection by extension, max-file-size skip.
**Acceptance**: `test_walker.py` proves correct file set on `fixtures/mini_app`,
including ignore rules.

### Phase 2 — Parse / symbol extraction
**Tasks**: `parsing/registry.py` maps extensions to grammars via
`tree-sitter-language-pack`. Write `.scm` query files to capture functions, classes,
methods, interfaces, and imports for the **target platform's language(s)**.
`extractor.py` runs queries and returns `Symbol` records (name, kind, path, lines).
**Acceptance**: `test_extractor_<lang>.py` extracts the expected symbols and imports
from fixtures. Emit a provisional `graph.json` with `File`/`Symbol` nodes and
`DECLARES` edges.

### Phase 3 — Resolution
**Tasks**: `resolver.py`. First pass: resolve `IMPORTS` exactly (module path → File
node). Second pass: resolve `CALLS` best-effort (match call name to declared symbol;
prefer same-file, then imported modules; attach `confidence`). Add `EXTENDS` /
`IMPLEMENTS`.
**Acceptance**: `test_resolver.py` — import edges 100% correct on fixtures; call edges
present with confidence; no crashes on unresolved names (drop or mark low-confidence).

### Phase 4 — Emitters
**Tasks**: `json_emitter.py` (full graph), `markdown_emitter.py` (frontmatter docs
per module/symbol), `mermaid_emitter.py` (`dependencies.mmd`).
**Acceptance**: `test_emit.py` validates JSON shape against the schema, frontmatter
parses, and the Mermaid file is non-empty and references real nodes.

### Phase 5 — Flow extractor (target platform)
**Tasks**: `flows/base.py` ABC + the **one** platform module for `config.platform`.
Add `Screen` nodes and `NAVIGATES_TO` edges (with triggers). Parse declarative nav
files (XML/storyboard) where relevant. Extend the Markdown emitter with
`docs/screens/*.md` and the Mermaid emitter with `flows.mmd`.
**Acceptance**: `test_flows_<platform>.py` — expected screens and navigation edges
on a fixture app; limitations reflected as lower confidence.

### Phase 6 — Persistence (LadybugDB behind a `GraphStore` interface)
**Tasks**: in `store.py`, define a `GraphStore` ABC (`create_schema`, `add_nodes`,
`add_edges`, `query`, `upsert_vectors`, `vector_search`, `close`) so the engine is
swappable. Implement `LadybugGraphStore` against the `ladybug` package (`import
ladybug`; API mirrors Kuzu v0.11.3). DDL derived from the schema, bulk-load
nodes/edges, Cypher query helpers. Write the index to `.klit-flow/`.
**Acceptance**: round-trip test against the `GraphStore` interface — load graph, run a
Cypher query (e.g. "what imports X"), get expected rows. Swapping engines requires no
caller changes outside `store.py`.

### Phase 7 — Hybrid search
**Tasks**: `embeddings.py` (local sentence-transformers over symbol name + doc +
file path), `bm25.py`, `search.py` (RRF fusion). Persist vectors via the
`GraphStore` interface (LadybugDB's vector index) or a local file.
**Acceptance**: `test_search.py` — a natural-language query returns the relevant
symbol ahead of distractors. No network calls (assert offline).

### Phase 8 — MCP server
**Tasks**: `server/mcp_server.py` with FastMCP. Tools:
- `query(text)` — hybrid search results.
- `context(symbol)` — inbound/outbound edges for a symbol.
- `impact(symbol)` — upstream dependents (graph traversal).
- `flows(screen?)` — navigation paths to/from a screen.
- `cypher(q)` — raw Cypher passthrough.
Wire `klit-flow serve` to launch it over stdio.
**Acceptance**: a smoke test starts the server and each tool returns well-formed
results against a pre-indexed fixture.

### Phase 9 — Optional local NL summaries
**Tasks**: behind a `--summaries` flag, use the local Ollama client to write a
1–2 sentence description into each doc's body. Off by default; no-op if Ollama
absent.
**Acceptance**: with the flag and a local model present, docs gain a summary; without
it, behavior is unchanged and offline.

---

## 10. CLI reference (target surface)

```
klit-flow analyze <path> [--platform android|ios|react_native|flutter]
                         [--summaries] [--force]
klit-flow query "<text>"            # hybrid search from the terminal
klit-flow flows [<screen>]          # list NAVIGATES_TO edges; filter by screen name
klit-flow serve                     # start MCP server (stdio)
klit-flow status                    # index freshness for the repo
klit-flow clean                     # remove .klit-flow/ for the repo
```

---

## 11. Testing strategy

- Keep a tiny `fixtures/mini_app/` for the target platform with a handful of files,
  2–4 screens, and known import/call/navigation edges. Tests assert against it.
- Every phase ships with tests; CI = `ruff check && ruff format --check && pytest`.
- Add an **offline assertion** in the search/index tests (monkeypatch sockets to
  fail) to guarantee no accidental network calls.

---

## 12. Conventions and guardrails (follow throughout)

- **Determinism in indexing**: no LLM/network in `analyze`. Embeddings local only.
- **Typed everything**: pydantic models for config, nodes, edges. No bare dicts
  crossing module boundaries.
- **Exact vs best-effort**: `IMPORTS` must be exact; `CALLS`/`NAVIGATES_TO` carry
  honest confidence and never silently fabricate edges.
- **Fail soft on parse errors**: a file that fails to parse is logged and skipped,
  never crashes the run.
- **Small, reviewable commits** per phase; do not jump ahead.
- **No secrets, no telemetry, no outbound calls** anywhere in the codebase.
- Stop and request review at the end of each phase.

---

## Open decision to confirm before Phase 5

Set `--platform` to the app you're analyzing (android / ios / react_native /
flutter). Only that flow-extractor module needs to be implemented for v1; the others
remain stubs behind the ABC.
