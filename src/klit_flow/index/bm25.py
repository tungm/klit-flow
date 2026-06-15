"""BM25 lexical index over graph node texts.

Wraps ``rank-bm25``'s ``BM25Okapi`` with simple persistence (pickle) so the
index can be saved alongside the LadybugDB store and reloaded without
re-indexing.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi


class BM25Index:
    """Lexical index over the corpus of indexed ``GraphNode`` texts.

    Typical usage::

        idx = BM25Index()
        for node in nodes:
            idx.add(node.id, node_text(node))
        idx.build()          # must be called before search()
        hits = idx.search("authentication activity", k=5)

    Persistence::

        idx.save(path)
        idx2 = BM25Index.load(path)
    """

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._corpus: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._built: bool = False

    # ── Building ──────────────────────────────────────────────────────────────

    def add(self, node_id: str, text: str) -> None:
        """Append one document to the corpus.  Must be called before :meth:`build`."""
        self._ids.append(node_id)
        self._corpus.append(text)

    def build(self) -> None:
        """Tokenise and build the BM25 index.  Must be called after all :meth:`add` calls."""
        self._built = True
        if not self._corpus:
            return  # empty index; search() will return []
        tokenised = [doc.lower().split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenised)

    # ── Searching ─────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return up to *k* ``(node_id, score)`` pairs, ranked by BM25 score.

        Parameters
        ----------
        query:
            Natural-language or keyword query string.
        k:
            Maximum number of results to return.

        Returns
        -------
        list[tuple[str, float]]
            Pairs of ``(node_id, bm25_score)``, descending by score.
            Nodes with a score of zero are excluded.
        """
        if not self._built:
            raise RuntimeError("BM25Index.build() must be called before search().")
        if self._bm25 is None:  # built but empty corpus
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            ((nid, float(score)) for nid, score in zip(self._ids, scores) if score > 0),
            key=lambda x: -x[1],
        )
        return ranked[:k]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Pickle the index to *path*."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "ids": self._ids,
                    "corpus": self._corpus,
                    "bm25": self._bm25,
                    "built": self._built,
                },
                fh,
            )

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        """Load a pickled index from *path*."""
        with path.open("rb") as fh:
            data = pickle.load(fh)  # noqa: S301 — internal, trusted path
        obj = cls()
        obj._ids = data["ids"]
        obj._corpus = data["corpus"]
        obj._bm25 = data["bm25"]
        obj._built = data.get("built", True)
        return obj
