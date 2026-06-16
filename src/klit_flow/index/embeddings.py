"""Local embedding model wrapper.

Uses ``sentence-transformers`` with the ``BAAI/bge-small-en-v1.5`` model by
default (384-dim, CPU-friendly, no network required after the first download).

The ``Embedder`` class is deliberately thin so it can be replaced with any
object that exposes the same ``dim`` property and ``encode()`` method.  Tests
inject a ``_FakeEmbedder`` to avoid requiring a GPU or model download.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder:
    """Wraps a ``SentenceTransformer`` model for local inference.

    Parameters
    ----------
    model_name:
        HuggingFace model id or local path.  Defaults to
        ``BAAI/bge-small-en-v1.5`` (384-dim, Apache 2.0 licence).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for semantic search. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        logger.debug("Loading embedding model %r", model_name)
        self._model = SentenceTransformer(model_name)
        get_dim = (
            getattr(self._model, "get_embedding_dimension", None)
            or self._model.get_sentence_embedding_dimension
        )
        self._dim: int = get_dim()

    @property
    def dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        return self._dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per text (normalised to unit length).

        Parameters
        ----------
        texts:
            Strings to embed.  May be a single-element list for query
            encoding.

        Returns
        -------
        list[list[float]]
            One ``float`` list per input text.
        """
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()
