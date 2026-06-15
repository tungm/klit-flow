"""Hybrid search: BM25 + semantic (vector) search fused with Reciprocal Rank Fusion.

Public API
----------
``node_text(node)``     — canonical text representation of a ``GraphNode``.
``build_index(...)``    — encode nodes, populate BM25 + vector store.
``hybrid_search(...)``  — run a query, return ranked node IDs.

Reciprocal Rank Fusion (RRF)
----------------------------
Each ranked list contributes ``1 / (k + rank)`` to a shared score map
(``k = 60`` is the standard constant from the original RRF paper).  The
two lists — BM25 and vector — are fused before taking the top-*k*.

All computation happens locally after the index is built; no network calls
are made during search.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from klit_flow.graph.model import GraphNode
from klit_flow.graph.store import GraphStore
from klit_flow.index.bm25 import BM25Index

if TYPE_CHECKING:
    from klit_flow.index.embeddings import Embedder

# RRF constant — standard value from the original paper.
_RRF_K = 60

# Splits camelCase / PascalCase so BM25 can match sub-words.
# "AuthActivity" → "Auth Activity"; "getBaseUrl" → "get Base Url"
_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _split_camel(s: str) -> str:
    return _CAMEL_RE.sub(" ", s)


def node_text(node: GraphNode) -> str:
    """Return the text string used to represent *node* in both indices.

    Format: ``<kind> <camel-split name> <file path>``

    Example: ``"Screen Auth Activity /src/AuthActivity.kt"``
    """
    split_name = _split_camel(node.name)
    return f"{node.kind} {split_name} {node.file_path}"


# ── RRF fusion ────────────────────────────────────────────────────────────────


def _rrf_score(rank: int) -> float:
    return 1.0 / (_RRF_K + rank)


def _rrf_fuse(
    bm25_results: list[tuple[str, float]],
    semantic_results: list[str],
    k: int,
) -> list[str]:
    """Fuse two ranked lists via RRF and return up to *k* node IDs."""
    scores: dict[str, float] = {}
    for rank, (nid, _) in enumerate(bm25_results):
        scores[nid] = scores.get(nid, 0.0) + _rrf_score(rank)
    for rank, nid in enumerate(semantic_results):
        scores[nid] = scores.get(nid, 0.0) + _rrf_score(rank)
    return sorted(scores, key=lambda nid: -scores[nid])[:k]


# ── Index building ────────────────────────────────────────────────────────────


def build_index(
    nodes: list[GraphNode],
    store: GraphStore,
    embedder: Embedder,
) -> BM25Index:
    """Index *nodes* into both BM25 and the vector store.

    Parameters
    ----------
    nodes:
        All ``GraphNode`` objects to index.
    store:
        Open ``GraphStore`` — vectors are persisted via
        :meth:`~klit_flow.graph.store.GraphStore.upsert_vectors`.
    embedder:
        Any object with ``dim: int`` and ``encode(texts) → list[list[float]]``.

    Returns
    -------
    BM25Index
        A built (ready-to-search) BM25 index.  Callers may persist it via
        :meth:`~klit_flow.index.bm25.BM25Index.save`.
    """
    if not nodes:
        bm25 = BM25Index()
        bm25.build()
        return bm25

    texts = [node_text(n) for n in nodes]

    # BM25
    bm25 = BM25Index()
    for node, text in zip(nodes, texts):
        bm25.add(node.id, text)
    bm25.build()

    # Semantic: encode all texts, persist to vector store
    vectors = embedder.encode(texts)
    store.upsert_vectors([n.id for n in nodes], vectors)

    return bm25


# ── Searching ─────────────────────────────────────────────────────────────────


def hybrid_search(
    query: str,
    bm25: BM25Index,
    store: GraphStore,
    embedder: Embedder,
    k: int = 10,
) -> list[str]:
    """Run a hybrid BM25 + semantic search and return up to *k* node IDs.

    Parameters
    ----------
    query:
        Natural-language or keyword query string.
    bm25:
        A built :class:`~klit_flow.index.bm25.BM25Index`.
    store:
        Open ``GraphStore`` with a populated vector index.
    embedder:
        The same embedder used during :func:`build_index`.
    k:
        Maximum number of node IDs to return.

    Returns
    -------
    list[str]
        Node IDs ranked by fused RRF score, most relevant first.
    """
    # Retrieve 2k candidates from each source to give RRF enough signal.
    bm25_hits = bm25.search(query, k=k * 2)
    query_vec = embedder.encode([query])[0]
    semantic_hits = store.vector_search(query_vec, k=k * 2)
    return _rrf_fuse(bm25_hits, semantic_hits, k=k)
