"""Phase 7 acceptance tests: hybrid search (BM25 + semantic + RRF).

All tests run fully offline — no model download, no network calls.
The ``_FakeEmbedder`` returns deterministic vectors so tests are fast,
reproducible, and independent of PyTorch or the actual sentence-transformer
model.

The offline assertion monkeypatches ``socket.socket`` to raise ``OSError``,
verifying that search itself makes no network calls once the index is built.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from klit_flow.graph.model import GraphNode, NodeKind
from klit_flow.graph.store import LadybugGraphStore
from klit_flow.index.bm25 import BM25Index
from klit_flow.index.search import build_index, hybrid_search, node_text

# ---------------------------------------------------------------------------
# Fake embedder — no sentence-transformers, no PyTorch
# ---------------------------------------------------------------------------

# Mapping of canonical texts to predetermined 4-dim unit vectors.
# These are chosen so that "Auth Activity" is near [1, 0, 0, 0] and
# "Utils" is near [0, 1, 0, 0], making search assertions deterministic.
_EMBEDDINGS: dict[str, list[float]] = {
    "Screen Auth Activity /src/AuthActivity.kt": [1.0, 0.0, 0.0, 0.0],
    "Screen Main Activity /src/MainActivity.kt": [0.0, 0.5, 0.5, 0.0],
    "Class Utils /src/Utils.kt": [0.0, 1.0, 0.0, 0.0],
    "Screen Profile Activity /src/ProfileActivity.kt": [0.0, 0.0, 1.0, 0.0],
    # Query vectors
    "auth": [0.95, 0.05, 0.0, 0.0],
    "Auth Activity": [1.0, 0.0, 0.0, 0.0],
    "profile screen": [0.05, 0.0, 0.95, 0.0],
}
_DEFAULT_VEC = [0.25, 0.25, 0.25, 0.25]


class _FakeEmbedder:
    """Deterministic embedder that returns preset vectors for known texts."""

    dim = 4

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [_EMBEDDINGS.get(text, _DEFAULT_VEC) for text in texts]


# ---------------------------------------------------------------------------
# Shared fixture nodes
# ---------------------------------------------------------------------------

_AUTH = GraphNode(
    id="s_auth",
    kind=NodeKind.Screen,
    name="AuthActivity",
    file_path="/src/AuthActivity.kt",
    start_line=1,
    end_line=80,
    language="kotlin",
)
_MAIN = GraphNode(
    id="s_main",
    kind=NodeKind.Screen,
    name="MainActivity",
    file_path="/src/MainActivity.kt",
    start_line=1,
    end_line=100,
    language="kotlin",
)
_UTILS = GraphNode(
    id="c_utils",
    kind=NodeKind.Class,
    name="Utils",
    file_path="/src/Utils.kt",
    start_line=1,
    end_line=30,
    language="kotlin",
)
_PROFILE = GraphNode(
    id="s_profile",
    kind=NodeKind.Screen,
    name="ProfileActivity",
    file_path="/src/ProfileActivity.kt",
    start_line=1,
    end_line=60,
    language="kotlin",
)

_ALL_NODES = [_AUTH, _MAIN, _UTILS, _PROFILE]


@pytest.fixture()
def store(tmp_path: Path) -> LadybugGraphStore:
    s = LadybugGraphStore(tmp_path / "graph.db", emb_dim=4)
    s.create_schema()
    s.add_nodes(_ALL_NODES)
    yield s
    s.close()


@pytest.fixture()
def built_index(store: LadybugGraphStore) -> BM25Index:
    return build_index(_ALL_NODES, store, _FakeEmbedder())


# ---------------------------------------------------------------------------
# node_text
# ---------------------------------------------------------------------------


def test_node_text_contains_kind() -> None:
    assert "Screen" in node_text(_AUTH)


def test_node_text_splits_camel_case() -> None:
    text = node_text(_AUTH)
    assert "Auth" in text
    assert "Activity" in text


def test_node_text_contains_file_path() -> None:
    assert "/src/AuthActivity.kt" in node_text(_AUTH)


# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------


def test_bm25_exact_match_ranks_first() -> None:
    # Need ≥ 3 docs: BM25 IDF = 0 when exactly 50% of docs match the term.
    idx = BM25Index()
    idx.add("s_auth", "Screen Auth Activity /src/AuthActivity.kt")
    idx.add("c_utils", "Class Utils /src/Utils.kt")
    idx.add("s_main", "Screen Main Activity /src/MainActivity.kt")
    idx.build()
    hits = idx.search("auth", k=5)
    assert hits, "Expected at least one result"
    assert hits[0][0] == "s_auth"


def test_bm25_returns_zero_score_nodes_excluded() -> None:
    idx = BM25Index()
    idx.add("n1", "hello world")
    idx.add("n2", "foo bar baz")
    idx.build()
    hits = idx.search("hello", k=5)
    ids = [h[0] for h in hits]
    assert "n2" not in ids  # "hello" doesn't appear in n2


def test_bm25_build_required_before_search() -> None:
    idx = BM25Index()
    idx.add("n1", "hello")
    with pytest.raises(RuntimeError, match="build"):
        idx.search("hello")


def test_bm25_save_and_load(tmp_path: Path) -> None:
    idx = BM25Index()
    idx.add("n1", "auth activity screen")
    idx.add("n2", "utils class")
    idx.build()

    p = tmp_path / "bm25.pkl"
    idx.save(p)
    loaded = BM25Index.load(p)

    hits_orig = idx.search("auth")
    hits_loaded = loaded.search("auth")
    assert hits_orig == hits_loaded


# ---------------------------------------------------------------------------
# build_index
# ---------------------------------------------------------------------------


def test_build_index_returns_bm25(store: LadybugGraphStore) -> None:
    bm25 = build_index(_ALL_NODES, store, _FakeEmbedder())
    assert isinstance(bm25, BM25Index)


def test_build_index_empty_nodes(store: LadybugGraphStore) -> None:
    bm25 = build_index([], store, _FakeEmbedder())
    assert bm25.search("anything") == []


def test_build_index_stores_vectors(store: LadybugGraphStore) -> None:
    """After build_index, vector_search must return node IDs."""
    build_index(_ALL_NODES, store, _FakeEmbedder())
    # Query near AuthActivity's vector
    results = store.vector_search([1.0, 0.0, 0.0, 0.0], k=4)
    assert "s_auth" in results


# ---------------------------------------------------------------------------
# hybrid_search — relevance
# ---------------------------------------------------------------------------


def test_hybrid_search_auth_ranks_first(built_index: BM25Index, store: LadybugGraphStore) -> None:
    """BM25 + semantic should rank AuthActivity first for an 'auth' query."""
    results = hybrid_search("auth", built_index, store, _FakeEmbedder(), k=4)
    assert results, "Expected results"
    assert results[0] == "s_auth"


def test_hybrid_search_relevant_before_distractor(
    built_index: BM25Index, store: LadybugGraphStore
) -> None:
    """AuthActivity must rank ahead of Utils for 'auth activity' query."""
    results = hybrid_search("auth activity", built_index, store, _FakeEmbedder(), k=4)
    assert "s_auth" in results
    assert results.index("s_auth") < results.index("c_utils")


def test_hybrid_search_profile_query(built_index: BM25Index, store: LadybugGraphStore) -> None:
    """'profile screen' should surface ProfileActivity near the top."""
    results = hybrid_search("profile screen", built_index, store, _FakeEmbedder(), k=4)
    assert "s_profile" in results


def test_hybrid_search_respects_k(built_index: BM25Index, store: LadybugGraphStore) -> None:
    results = hybrid_search("activity", built_index, store, _FakeEmbedder(), k=2)
    assert len(results) <= 2


def test_hybrid_search_no_duplicates(built_index: BM25Index, store: LadybugGraphStore) -> None:
    results = hybrid_search("screen activity", built_index, store, _FakeEmbedder(), k=10)
    assert len(results) == len(set(results))


# ---------------------------------------------------------------------------
# Offline assertion
# ---------------------------------------------------------------------------


def test_hybrid_search_is_offline(
    built_index: BM25Index,
    store: LadybugGraphStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Querying an already-built index must not make any network calls."""

    def _no_network(*args: object, **kwargs: object) -> None:
        raise OSError("Network access is forbidden during offline search")

    monkeypatch.setattr(socket, "socket", _no_network)

    # Both BM25 and vector search are purely local — this must not raise.
    results = hybrid_search("auth", built_index, store, _FakeEmbedder(), k=4)
    assert results  # got results without touching the network


def test_bm25_search_is_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """BM25 search must work with sockets blocked."""
    idx = BM25Index()
    idx.add("s_auth", "Screen Auth Activity /src/AuthActivity.kt")
    idx.add("c_utils", "Class Utils /src/Utils.kt")
    idx.add("s_main", "Screen Main Activity /src/MainActivity.kt")
    idx.build()

    def _no_network(*args: object, **kwargs: object) -> None:
        raise OSError("no network")

    monkeypatch.setattr(socket, "socket", _no_network)
    hits = idx.search("auth", k=3)
    assert hits[0][0] == "s_auth"
