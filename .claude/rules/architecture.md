# Project Architecture Rules

## Overview

This project follows the **monorepo shared-core pattern**: three surfaces (CLI, Web portal, MCP server) backed by a single framework-agnostic core library. All business logic lives in `core/`; surfaces are thin adapters only.

```
src/<package>/
├── core/       # domain models, services, DB, config — NO framework imports
├── cli/        # Typer commands — calls core, never web/ or mcp/
├── web/        # FastAPI routes — calls core, never cli/ or mcp/
└── mcp/        # FastMCP tools/resources — calls core, never cli/ or web/
```

---

## Core Rules

### The Golden Rule: surfaces never import from each other
- `cli/`, `web/`, and `mcp/` may only import from `core/`.
- Cross-surface imports (`cli` importing from `web`, etc.) are **never allowed**.
- If logic is needed by more than one surface, it belongs in `core/`.

### `core/` must stay framework-agnostic
- No FastAPI, Typer, Click, or FastMCP imports inside `core/`.
- No HTTP request/response objects in core service signatures.
- No CLI argument parsing in core.
- Allowed in core: Pydantic, SQLAlchemy/SQLModel, `pydantic-settings`, standard library.

### Pydantic models are the contract
- All data crossing a surface boundary (in or out of core) must be typed with Pydantic models.
- Never pass raw dicts between a surface and core services.
- Shared schemas live in `core/schemas.py` or `core/models.py`.

---

## Directory Conventions

### `core/`
| File | Purpose |
|------|---------|
| `models.py` | SQLAlchemy / SQLModel ORM models |
| `schemas.py` | Pydantic request/response schemas |
| `services.py` | Business logic functions (or split into `services/` package) |
| `db.py` | Database engine, session factory, migrations entry |
| `config.py` | `pydantic-settings` Settings class — single source of config |
| `exceptions.py` | Domain-level exception classes |

### `cli/`
| File | Purpose |
|------|---------|
| `__init__.py` | Exports the root Typer `app` |
| `commands/` | One module per command group |
| `formatters.py` | Rich / plain-text output helpers |

### `web/`
| File | Purpose |
|------|---------|
| `main.py` | FastAPI app factory (`create_app()`) |
| `routers/` | One router per domain area |
| `dependencies.py` | FastAPI `Depends()` helpers (DB session, auth, etc.) |
| `middleware.py` | CORS, logging, error-handling middleware |

### `mcp/`
| File | Purpose |
|------|---------|
| `server.py` | FastMCP server instance and entry point |
| `tools/` | One module per logical tool group |
| `resources/` | MCP resource handlers |

---

## Coding Standards

### General
- Python 3.11+ minimum; use modern type hints (`list[str]` not `List[str]`).
- All public functions and classes must have docstrings.
- Use `pathlib.Path` over `os.path` everywhere.
- Prefer `async`/`await` in `web/` and `mcp/`; keep `core/` services sync-compatible where possible, or provide both sync and async variants.

### Configuration
- All config is read from environment variables via the `Settings` class in `core/config.py`.
- Never hardcode secrets, URLs, or environment-specific values anywhere.
- Surfaces access config by importing `from core.config import get_settings`.

### Error handling
- Core raises domain exceptions from `core/exceptions.py`.
- Each surface is responsible for catching domain exceptions and translating them to the appropriate surface-level error (HTTP status code, CLI exit code, MCP error response).
- Never let raw SQLAlchemy or DB errors bubble up to a surface uncaught.

### Database
- Session lifecycle is managed at the surface boundary, not inside core services.
- Core service functions accept a session as an argument — they do not create sessions.
- Web: use `Depends(get_db)`. CLI: use a context manager. MCP: use lifespan context.

---

## Toolchain

| Concern | Tool |
|---------|------|
| CLI framework | Typer |
| Web framework | FastAPI + Uvicorn |
| MCP framework | FastMCP |
| ORM | SQLModel (or SQLAlchemy 2.x + Pydantic) |
| Config | pydantic-settings |
| Testing | pytest + pytest-asyncio |
| Linting | Ruff |
| Type checking | pyright or mypy |
| Packaging | pyproject.toml (Hatch or uv) |

---

## Entry Points (`pyproject.toml`)

```toml
[project.scripts]
mytool     = "mypackage.cli:app"       # Typer root app
mytool-web = "mypackage.web.main:main" # uvicorn launcher
mytool-mcp = "mypackage.mcp.server:main" # FastMCP stdio entry
```

---

## Testing Rules

- `core/` tests are pure unit tests — no HTTP client, no CLI runner, no MCP client.
- `web/` tests use FastAPI `TestClient` or `AsyncClient`; mock core services at the router boundary.
- `cli/` tests use Typer's `CliRunner`; mock core services.
- `mcp/` tests use FastMCP's test utilities; mock core services.
- Aim for high coverage in `core/`; surface tests verify wiring, not business logic.
- Test file mirrors source: `tests/core/`, `tests/cli/`, `tests/web/`, `tests/mcp/`.

---

## What Goes Where — Quick Reference

| Scenario | Location |
|----------|----------|
| New business rule or calculation | `core/services.py` |
| New database table | `core/models.py` + migration |
| New Pydantic schema shared by surfaces | `core/schemas.py` |
| New CLI command | `cli/commands/<group>.py` |
| New REST endpoint | `web/routers/<domain>.py` |
| New MCP tool exposed to AI agents | `mcp/tools/<group>.py` |
| New MCP resource | `mcp/resources/<name>.py` |
| New config value | `core/config.py` Settings class + `.env.example` |
| Shared formatting / output util | Only if used by 2+ surfaces → `core/utils.py`; otherwise keep in that surface |

---

## Anti-Patterns to Avoid

- ❌ Importing `fastapi`, `typer`, or `fastmcp` inside `core/`
- ❌ Putting SQL queries directly in route handlers or CLI commands
- ❌ Creating DB sessions inside core service functions
- ❌ Hardcoding config values or using `os.environ.get()` outside `core/config.py`
- ❌ Sharing mutable global state between surfaces
- ❌ Duplicating a service function across two surfaces instead of moving it to `core/`
- ❌ Returning ORM model objects directly from web routes (use schemas)
- ❌ Raising `HTTPException` inside `core/` (that's a FastAPI concern)
