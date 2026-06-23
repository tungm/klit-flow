"""User-defined named screen flows.

A *named flow* is a human-curated set of screen paths grouped under one label
(e.g. "Login flow").  A flow may **branch**: it holds one or more *branches*,
each an ordered, linear sequence of screens.  Branches typically share a common
prefix — for example a "Checkout" flow might contain::

    A -> B -> C1 -> D     (branch 1)
    A -> B -> C2          (branch 2)

Unlike the auto-extracted ``NAVIGATES_TO`` edges, named flows are created and
edited by users in the web portal and persisted to a JSON file inside the
target repo's ``.klit-flow/`` directory.

Persistence is intentionally a plain JSON file (``named_flows.json``) rather
than the graph DB: the ``analyze`` pipeline deletes and recreates the graph DB
on ``--force``, and user-authored flows must survive re-indexing.

Search semantics: a flow matches a queried screen sequence when the query is an
**ordered subsequence** (gaps allowed) of *any one branch's* screen names,
compared case-insensitively.  For example the query ``B -> C`` matches both
``A -> B -> C`` and ``B -> C -> D``, and the query ``A -> C`` matches
``A -> B -> C``.

This module is framework-agnostic: it knows nothing about FastAPI, the graph
store, or navigation-edge validation.  Surfaces (e.g. the web portal) are
responsible for validating that consecutive screens within each branch are
connected by real ``NAVIGATES_TO`` edges before calling
:meth:`NamedFlowStore.create`.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class NamedFlowScreen(BaseModel):
    """One ordered step in a branch."""

    id: str
    name: str


class NamedFlowBranch(BaseModel):
    """One linear path within a named flow."""

    label: str = ""
    screens: list[NamedFlowScreen] = Field(default_factory=list)


class NamedFlow(BaseModel):
    """A user-curated, possibly branching, collection of screen paths."""

    id: str
    name: str
    branches: list[NamedFlowBranch] = Field(default_factory=list)
    created_at: str
    updated_at: str

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy(cls, data: object) -> object:
        """Upgrade pre-branch records (a flat ``screens`` list) to a single branch."""
        if isinstance(data, dict) and "branches" not in data and "screens" in data:
            data = {**data, "branches": [{"screens": data.get("screens") or []}]}
            data.pop("screens", None)
        return data


def _now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def _is_ordered_subsequence(needle: list[str], haystack: list[str]) -> bool:
    """Return True if *needle* appears in *haystack* in order, gaps allowed.

    Comparison is case-insensitive.  An empty *needle* never matches (callers
    should treat an empty query as "no filter" themselves).
    """
    if not needle:
        return False
    lower_needle = [s.lower() for s in needle]
    it = iter(s.lower() for s in haystack)
    return all(any(h == n for h in it) for n in lower_needle)


def _flow_matches(flow: NamedFlow, sequence: list[str]) -> bool:
    """True if *sequence* is an ordered subsequence of any branch in *flow*."""
    return any(
        _is_ordered_subsequence(sequence, [s.name for s in branch.screens])
        for branch in flow.branches
    )


class NamedFlowStore:
    """JSON-file-backed CRUD store for :class:`NamedFlow` records.

    Parameters
    ----------
    path:
        Location of the backing ``named_flows.json`` file.  When ``None`` the
        store is purely in-memory (useful for tests); mutations are not
        persisted.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._flows: dict[str, NamedFlow] = {}
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for item in raw.get("flows", []):
            try:
                flow = NamedFlow.model_validate(item)
            except ValueError:
                continue
            self._flows[flow.id] = flow

    def _save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"flows": [f.model_dump() for f in self._flows.values()]}
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ── Queries ────────────────────────────────────────────────────────────

    def list(self) -> list[NamedFlow]:
        """Return all named flows, most recently created/updated first.

        Ordering follows insertion order (newest last); :meth:`update` moves a
        flow to the end so edited flows surface first.  This is independent of
        timestamp resolution, which can tie for flows created in the same tick.
        """
        return list(reversed(self._flows.values()))

    def get(self, flow_id: str) -> NamedFlow | None:
        """Return the flow with *flow_id*, or ``None`` if absent."""
        return self._flows.get(flow_id)

    def search(self, sequence: list[str]) -> list[NamedFlow]:
        """Return flows in which *sequence* is an ordered subsequence of some branch.

        *sequence* is a list of screen names. Matching is case-insensitive and
        allows gaps between the queried screens. An empty *sequence* returns all
        flows (treated as "no filter").
        """
        if not sequence:
            return self.list()
        return [f for f in self.list() if _flow_matches(f, sequence)]

    # ── Mutations ──────────────────────────────────────────────────────────

    def create(self, name: str, branches: list[NamedFlowBranch]) -> NamedFlow:
        """Create and persist a new named flow."""
        now = _now()
        flow = NamedFlow(
            id=uuid.uuid4().hex,
            name=name,
            branches=branches,
            created_at=now,
            updated_at=now,
        )
        self._flows[flow.id] = flow
        self._save()
        return flow

    def update(
        self,
        flow_id: str,
        *,
        name: str | None = None,
        branches: list[NamedFlowBranch] | None = None,
    ) -> NamedFlow | None:
        """Update a flow's name and/or branches. Returns the updated flow or ``None``."""
        flow = self._flows.get(flow_id)
        if flow is None:
            return None
        if name is not None:
            flow.name = name
        if branches is not None:
            flow.branches = branches
        flow.updated_at = _now()
        self._flows[flow_id] = self._flows.pop(flow_id)  # move to end (most recent)
        self._save()
        return flow

    def delete(self, flow_id: str) -> bool:
        """Delete a flow. Returns ``True`` if a flow was removed."""
        if flow_id in self._flows:
            del self._flows[flow_id]
            self._save()
            return True
        return False
