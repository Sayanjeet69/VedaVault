"""Local vector store contract and NumPy-backed cosine similarity implementation."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .documents import RetrievalDocument
from .embeddings import EmbeddingConfiguration, EmbeddingProvider
from .filters import MetadataFilter, matches_metadata

if TYPE_CHECKING:
    from collections.abc import Mapping


INDEX_MANIFEST_FILENAME = "index_manifest.json"


class IndexCompatibilityError(ValueError):
    """Raised when an embedding provider cannot safely query a persisted index."""


class IndexManifestError(ValueError):
    """Raised when a persisted local index has no usable manifest."""


class IndexManifest:
    """Versioned local-index metadata required to prevent semantic mismatches."""

    SCHEMA_VERSION = 1

    def __init__(self, embedding: EmbeddingConfiguration) -> None:
        self.embedding = embedding

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.SCHEMA_VERSION, "embedding": self.embedding.to_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IndexManifest":
        if set(value) != {"schema_version", "embedding"} or value["schema_version"] != cls.SCHEMA_VERSION:
            raise IndexManifestError("index manifest has an unsupported schema; rebuild the local index")
        if not isinstance(value["embedding"], dict):
            raise IndexManifestError("index manifest embedding configuration is invalid; rebuild the local index")
        try:
            return cls(EmbeddingConfiguration.from_dict(value["embedding"]))
        except ValueError as exc:
            raise IndexManifestError(f"index manifest is invalid; rebuild the local index ({exc})") from exc


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
        self._manifest: IndexManifest | None = None

    @property
    def size(self) -> int:
        return len(self._documents)

    @property
    def manifest(self) -> IndexManifest | None:
        return self._manifest

    @property
    def embedding_dimension(self) -> int | None:
        return None if self._vectors is None else int(self._vectors.shape[1])

    def set_manifest(self, manifest: IndexManifest) -> None:
        if self.embedding_dimension is None:
            raise ValueError("cannot set an index manifest before adding vectors")
        if manifest.embedding.embedding_dimension != self.embedding_dimension:
            raise ValueError("index manifest embedding dimension does not match vectors")
        self._manifest = manifest

    def validate_embedding_provider(self, provider: EmbeddingProvider) -> None:
        if self._manifest is None:
            raise IndexManifestError("index has no manifest; rebuild it before querying")
        configuration = provider.index_configuration(self._manifest.embedding.embedding_dimension)
        if configuration is None:
            raise IndexCompatibilityError("embedding provider does not expose a persisted configuration for this index")
        if configuration != self._manifest.embedding:
            raise IndexCompatibilityError(
                "embedding configuration is incompatible with this index; use the indexed model/profile "
                "or rebuild the index with the requested configuration"
            )

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
        if self._manifest is None:
            raise IndexManifestError("cannot save an index without an embedding manifest")
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "vectors.npz", vectors=self._vectors)
        (directory / "documents.json").write_text(
            json.dumps([document.to_dict() for document in self._documents], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (directory / INDEX_MANIFEST_FILENAME).write_text(
            json.dumps(self._manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path, embedding_provider: EmbeddingProvider | None = None) -> "LocalVectorStore":
        documents = [RetrievalDocument.from_dict(value) for value in json.loads((directory / "documents.json").read_text(encoding="utf-8"))]
        vectors = np.load(directory / "vectors.npz")["vectors"].astype(np.float32)
        if len(documents) != len(vectors):
            raise ValueError("stored document and vector counts do not match")
        manifest_path = directory / INDEX_MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise IndexManifestError("index manifest is missing; rebuild the local index")
        try:
            manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise IndexManifestError("index manifest is not valid JSON; rebuild the local index") from exc
        if not isinstance(manifest_value, dict):
            raise IndexManifestError("index manifest must be a JSON object; rebuild the local index")
        manifest = IndexManifest.from_dict(manifest_value)
        if vectors.ndim != 2 or manifest.embedding.embedding_dimension != vectors.shape[1]:
            raise IndexManifestError("index manifest embedding dimension does not match vectors; rebuild the local index")
        store = cls()
        store._documents = documents
        store._vectors = vectors
        store._manifest = manifest
        if embedding_provider is not None:
            store.validate_embedding_provider(embedding_provider)
        return store


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, np.finfo(np.float32).eps)
