"""Embedding provider contract and the local Sentence Transformers adapter."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class EmbeddingProvider(ABC):
    """An interchangeable text-to-vector provider."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return one float vector per input text."""


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Lazy local adapter; importing this package does not load a model."""

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        self.model_name = model_name or os.getenv("VEDAVAULT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.device = device or os.getenv("VEDAVAULT_EMBEDDING_DEVICE")
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for local semantic retrieval. "
                    "Install Backend/requirements-retrieval.txt."
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self._get_model().encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)
