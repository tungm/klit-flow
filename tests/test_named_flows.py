"""Tests for the user-defined named-flow store and ordered-subsequence search."""

from __future__ import annotations

from pathlib import Path

from klit_flow.named_flows import (
    NamedFlow,
    NamedFlowBranch,
    NamedFlowScreen,
    NamedFlowStore,
    _is_ordered_subsequence,
)


def _branch(*names: str) -> NamedFlowBranch:
    return NamedFlowBranch(screens=[NamedFlowScreen(id=n.lower(), name=n) for n in names])


def _one(*names: str) -> list[NamedFlowBranch]:
    """A single-branch flow body."""
    return [_branch(*names)]


# ---------------------------------------------------------------------------
# Ordered-subsequence matching
# ---------------------------------------------------------------------------


def test_subsequence_consecutive() -> None:
    assert _is_ordered_subsequence(["B", "C"], ["A", "B", "C"])
    assert _is_ordered_subsequence(["B", "C"], ["B", "C", "D"])


def test_subsequence_with_gaps() -> None:
    assert _is_ordered_subsequence(["A", "C"], ["A", "B", "C"])


def test_subsequence_wrong_order() -> None:
    assert not _is_ordered_subsequence(["C", "B"], ["A", "B", "C"])


def test_subsequence_missing_screen() -> None:
    assert not _is_ordered_subsequence(["A", "X"], ["A", "B", "C"])


def test_subsequence_case_insensitive() -> None:
    assert _is_ordered_subsequence(["b", "c"], ["A", "B", "C"])


def test_subsequence_empty_needle_never_matches() -> None:
    assert not _is_ordered_subsequence([], ["A", "B"])


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def test_create_and_get() -> None:
    store = NamedFlowStore()
    flow = store.create("Login", _one("A", "B", "C"))
    assert flow.id
    got = store.get(flow.id)
    assert got is not None
    assert [s.name for s in got.branches[0].screens] == ["A", "B", "C"]


def test_list_orders_recent_first() -> None:
    store = NamedFlowStore()
    first = store.create("First", _one("A", "B"))
    second = store.create("Second", _one("B", "C"))
    ids = [f.id for f in store.list()]
    # Most recently updated first
    assert ids.index(second.id) < ids.index(first.id)


def test_update_name_and_branches() -> None:
    store = NamedFlowStore()
    flow = store.create("Old", _one("A", "B"))
    updated = store.update(flow.id, name="New", branches=_one("A", "B", "C"))
    assert updated is not None
    assert updated.name == "New"
    assert [s.name for s in updated.branches[0].screens] == ["A", "B", "C"]


def test_update_missing_returns_none() -> None:
    store = NamedFlowStore()
    assert store.update("nope", name="x") is None


def test_delete() -> None:
    store = NamedFlowStore()
    flow = store.create("Login", _one("A", "B"))
    assert store.delete(flow.id) is True
    assert store.get(flow.id) is None
    assert store.delete(flow.id) is False


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_search_matches_multiple_flows() -> None:
    store = NamedFlowStore()
    store.create("flow1", _one("A", "B", "C"))
    store.create("flow2", _one("B", "C", "D"))
    store.create("other", _one("X", "Y"))
    names = {f.name for f in store.search(["B", "C"])}
    assert names == {"flow1", "flow2"}


def test_search_with_gap() -> None:
    store = NamedFlowStore()
    store.create("flow1", _one("A", "B", "C"))
    assert [f.name for f in store.search(["A", "C"])] == ["flow1"]


def test_search_empty_returns_all() -> None:
    store = NamedFlowStore()
    store.create("flow1", _one("A", "B"))
    store.create("flow2", _one("C", "D"))
    assert len(store.search([])) == 2


def test_create_allows_repeated_screen() -> None:
    store = NamedFlowStore()
    flow = store.create("Loop", _one("A", "B", "A"))
    assert [s.name for s in flow.branches[0].screens] == ["A", "B", "A"]
    # A repeated query matches because A occurs twice in order.
    assert [f.name for f in store.search(["A", "A"])] == ["Loop"]


def test_update_moves_flow_to_front() -> None:
    store = NamedFlowStore()
    first = store.create("First", _one("A", "B"))
    store.create("Second", _one("B", "C"))
    # Editing the older flow should surface it first in the listing.
    store.update(first.id, name="First (edited)")
    assert store.list()[0].id == first.id


# ---------------------------------------------------------------------------
# Branches
# ---------------------------------------------------------------------------


def test_search_matches_either_branch() -> None:
    store = NamedFlowStore()
    # A -> B -> C1 -> D  and  A -> B -> C2
    store.create("Checkout", [_branch("A", "B", "C1", "D"), _branch("A", "B", "C2")])
    assert [f.name for f in store.search(["C1", "D"])] == ["Checkout"]
    assert [f.name for f in store.search(["B", "C2"])] == ["Checkout"]
    # Shared prefix matches via either branch.
    assert [f.name for f in store.search(["A", "B"])] == ["Checkout"]


def test_search_no_match_across_branches() -> None:
    store = NamedFlowStore()
    store.create("Checkout", [_branch("A", "B", "C1", "D"), _branch("A", "B", "C2")])
    # C1 and C2 are in different branches; a sequence spanning both must not match.
    assert store.search(["C1", "C2"]) == []


def test_branch_is_searched_independently() -> None:
    store = NamedFlowStore()
    store.create("Multi", [_branch("A", "B"), _branch("X", "Y", "Z")])
    assert [f.name for f in store.search(["X", "Z"])] == ["Multi"]
    assert store.search(["B", "X"]) == []


# ---------------------------------------------------------------------------
# Persistence & legacy migration
# ---------------------------------------------------------------------------


def test_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / ".klit-flow" / "named_flows.json"
    store = NamedFlowStore(path)
    flow = store.create("Checkout", [_branch("A", "B", "C1", "D"), _branch("A", "B", "C2")])

    reopened = NamedFlowStore(path)
    again = reopened.get(flow.id)
    assert again is not None
    assert again.name == "Checkout"
    assert len(again.branches) == 2
    assert [s.name for s in again.branches[0].screens] == ["A", "B", "C1", "D"]


def test_corrupt_file_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "named_flows.json"
    path.write_text("{ not valid json", encoding="utf-8")
    store = NamedFlowStore(path)  # must not raise
    assert store.list() == []


def test_legacy_flat_screens_are_migrated_to_one_branch() -> None:
    # Pre-branch records stored a flat ``screens`` list with no ``branches``.
    legacy = {
        "id": "abc",
        "name": "Old flow",
        "screens": [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
        "created_at": "2020-01-01T00:00:00+00:00",
        "updated_at": "2020-01-01T00:00:00+00:00",
    }
    flow = NamedFlow.model_validate(legacy)
    assert len(flow.branches) == 1
    assert [s.name for s in flow.branches[0].screens] == ["A", "B"]
