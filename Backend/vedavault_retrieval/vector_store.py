"""Local vector store contract and NumPy-backed cosine similarity implementation."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .documents import RetrievalDocument
from .filters import MetadataFilter, matches_metadata


class VectorStore(ABC):
    @abstractmethod
    def add(self, documents: list[RetrievalDocument], vectors: np.ndarray) -> None:
        """Add documents and their corresponding vectors."""

    @abstractmethod
    def search(self, query_vector: np.ndarray, limit: int, filters: MetadataFilter | None = None) -> list[tuple[RetrievalDocument, float]]:
        """Return matching documents ordered by descending similarity."""


class LocalVectorStore(VectorStore):
    """Small, portable cosine-similarity vector store suitable for local development."""

    def __init__(self) -> None:
        self._documents: list[RetrievalDocument] = []
        self._vectors: np.ndarray | None = None

    @property
    def size(self) -> int:
        return len(self._documents)

    def add(self, documents: list[RetrievalDocument], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if not documents:
            return
        if vectors.ndim != 2 or vectors.shape[0] != len(documents):
            raise ValueError("vectors must be a two-dimensional array with one row per document")
        if self._vectors is not None and vectors.shape[1] != self._vectors.shape[1]:
            raise ValueError("embedding dimensions do not match existing index")
        if len({document.document_id for document in documents}) != len(documents):
            raise ValueError("document IDs must be unique within an add operation")
        if {document.document_id for document in documents} & {document.document_id for document in self._documents}:
            raise ValueError("document ID already exists in index")
        normalized = _normalize(vectors)
        self._vectors = normalized if self._vectors is None else np.vstack((self._vectors, normalized))
        self._documents.extend(documents)

    def search(self, query_vector: np.ndarray, limit: int, filters: MetadataFilter | None = None) -> list[tuple[RetrievalDocument, float]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        if self._vectors is None:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self._vectors.shape[1]:
            raise ValueError("query embedding dimension does not match index")
        scores = self._vectors @ _normalize(query.reshape(1, -1))[0]
        ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)
        return [
            (self._documents[index], float(score))
            for index, score in ranked
            if matches_metadata(self._documents[index].metadata, filters)
        ][:limit]

    def save(self, directory: Path) -> None:
        if self._vectors is None:
            raise ValueError("cannot save an empty vector store")
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "vectors.npz", vectors=self._vectors)
        (directory / "documents.json").write_text(
            json.dumps([document.to_dict() for document in self._documents], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "LocalVectorStore":
        documents = [RetrievalDocument.from_dict(value) for value in json.loads((directory / "documents.json").read_text(encoding="utf-8"))]
        vectors = np.load(directory / "vectors.npz")["vectors"].astype(np.float32)
        if len(documents) != len(vectors):
            raise ValueError("stored document and vector counts do not match")
        store = cls()
        store._documents = documents
        store._vectors = vectors
        return store


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, np.finfo(np.float32).eps)
