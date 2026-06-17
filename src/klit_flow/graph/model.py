"""Authoritative graph schema for klit-flow.

The LadybugDB DDL (Phase 6) and JSON emitter (Phase 4) derive from these models.
All graph persistence must use GraphNode / GraphEdge — no bare dicts across boundaries.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class NodeKind(StrEnum):
    File = "File"
    Module = "Module"
    Function = "Function"
    Class = "Class"
    Method = "Method"
    Interface = "Interface"
    Screen = "Screen"


class RelationType(StrEnum):
    DECLARES = "DECLARES"
    IMPORTS = "IMPORTS"
    CALLS = "CALLS"
    EXTENDS = "EXTENDS"
    IMPLEMENTS = "IMPLEMENTS"
    NAVIGATES_TO = "NAVIGATES_TO"


class GraphNode(BaseModel):
    id: str
    kind: NodeKind
    name: str
    file_path: str
    start_line: int
    end_line: int
    language: str


class ConditionLevel(BaseModel):
    """One level in a nested condition chain.

    Each level represents a single control-flow guard enclosing a navigation
    call.  Levels are ordered outermost-first (the first item is the
    top-level enclosing condition).
    """

    expression: str  # e.g. "loginResult is Success", "user.isAdmin"
    kind: str  # "if" | "else" | "when_branch" | "when_else" | "annotation"
    source_line: int | None = None


class GraphEdge(BaseModel):
    src_id: str
    dst_id: str
    type: RelationType
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    file_path: str | None = None
    line: int | None = None
    trigger: str | None = None  # NAVIGATES_TO: e.g. "button_tap", "api_response"
    conditions: list[ConditionLevel] = Field(default_factory=list)
