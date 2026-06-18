"""Local embedding model wrapper.

Uses ``sentence-transformers`` with the ``BAAI/bge-small-en-v1.5`` model by
default (384-dim, CPU-friendly, no network required after the first download).

The ``Embedder`` class is deliberately thin so it can be replaced with any
object that exposes the same ``dim`` property and ``encode()`` method.  Tests
inject a ``_FakeEmbedder`` to avoid requiring a GPU or model download.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# If KLIT_FLOW_MODEL_DIR is set, the Embedder loads the model from that local
# directory instead of downloading it from HuggingFace.  Point it at the
# bundled models/ folder from a release archive for fully offline operation:
#
#   KLIT_FLOW_MODEL_DIR=/path/to/release/v1.0.0/models/bge-small-en-v1.5
_MODEL_DIR_ENV = "KLIT_FLOW_MODEL_DIR"


def _resolve_model(model_name: str) -> tuple[str, bool]:
    """Return (model_path_or_id, local_only).

    When KLIT_FLOW_MODEL_DIR is set and points to a valid directory, return
    the expanded local path with ``local_only=True`` so sentence-transformers
    never contacts HuggingFace.  Otherwise return the model id with
    ``local_only=False``.
    """
    env = os.environ.get(_MODEL_DIR_ENV, "").strip()
    logger.warning("[klit-flow] %s = %r", _MODEL_DIR_ENV, env or "(not set)")

    if env:
        p = Path(env).expanduser().resolve()
        logger.warning("[klit-flow] model path resolved to: %s  exists=%s", p, p.is_dir())
        if p.is_dir():
            contents = [f.name for f in p.iterdir()] if p.is_dir() else []
            logger.warning("[klit-flow] model dir contents: %s", contents)
            return str(p), True
        logger.warning(
            "[klit-flow] KLIT_FLOW_MODEL_DIR %r resolved to %s which is not a directory; "
            "falling back to HuggingFace download",
            env,
            p,
        )
    else:
        logger.warning(
            "[klit-flow] KLIT_FLOW_MODEL_DIR not set; will download from HuggingFace. "
            "Set KLIT_FLOW_MODEL_DIR to use the bundled offline model."
        )
    return model_name, False


class Embedder:
    """Wraps a ``SentenceTransformer`` model for local inference.

    Parameters
    ----------
    model_name:
        HuggingFace model id or local path.  Defaults to
        ``BAAI/bge-small-en-v1.5`` (384-dim, Apache 2.0 licence).
        Overridden by the ``KLIT_FLOW_MODEL_DIR`` environment variable.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model = None
        self._dim: int = 384  # default; updated on successful load

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for semantic search. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        resolved, local_only = _resolve_model(model_name)
        logger.warning(
            "[klit-flow] loading embedding model resolved=%r local_files_only=%s",
            resolved,
            local_only,
        )
        try:
            self._model = SentenceTransformer(resolved, local_files_only=local_only)
            get_dim = (
                getattr(self._model, "get_embedding_dimension", None)
                or self._model.get_sentence_embedding_dimension
            )
            self._dim = get_dim()
            logger.warning("[klit-flow] embedding model loaded OK, dim=%d", self._dim)
        except OSError as exc:
            logger.warning(
                "[klit-flow] embedding model load FAILED (OSError): %s\n"
                "  resolved path: %r\n"
                "  local_files_only: %s\n"
                "  If 'header too large': a corrupted file exists at the path above. "
                "Delete it and re-run 'klit-flow download-model'.\n"
                "  If 'no local files found': KLIT_FLOW_MODEL_DIR does not contain model weights. "
                "Check the path above is the model directory (should contain model.safetensors).",
                exc,
                resolved,
                local_only,
            )
        except Exception as exc:
            logger.warning(
                "[klit-flow] embedding model load FAILED (%s): %s  resolved=%r",
                type(exc).__name__,
                exc,
                resolved,
            )

    @property
    def available(self) -> bool:
        """True if the model loaded successfully and encode() can be called."""
        return self._model is not None

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
        if self._model is None:
            raise RuntimeError(
                "Embedding model is not available. "
                "Check logs for the load error (likely an out-of-memory condition)."
            )
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()
