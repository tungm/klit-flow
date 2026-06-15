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


class GraphEdge(BaseModel):
    src_id: str
    dst_id: str
    type: RelationType
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    file_path: str | None = None
    line: int | None = None
    trigger: str | None = None  # NAVIGATES_TO: e.g. "button_tap", "deep_link"
