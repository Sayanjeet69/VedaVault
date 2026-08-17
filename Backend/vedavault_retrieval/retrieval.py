"""Retrieval orchestration over VedaVault-controlled embedding and index contracts."""

from __future__ import annotations

from dataclasses import dataclass

from .embeddings import EmbeddingProvider
from .filters import MetadataFilter
from .vector_store import VectorStore
from .documents import RetrievalDocument


@dataclass(frozen=True)
class RetrievalResult:
    document: RetrievalDocument
    score: float


class Retriever:
    def __init__(self, embedding_provider: EmbeddingProvider, vector_store: VectorStore) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        validate_provider = getattr(vector_store, "validate_embedding_provider", None)
        if getattr(vector_store, "manifest", None) is not None and validate_provider is not None:
            validate_provider(embedding_provider)

    def retrieve(self, query: str, limit: int = 5, filters: MetadataFilter | None = None) -> list[RetrievalResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if limit < 1:
            raise ValueError("limit must be positive")
        vector = self.embedding_provider.embed_query(query)
        if vector.ndim != 2 or vector.shape[0] != 1:
            raise ValueError("embedding provider must return one vector for a query")
        return [RetrievalResult(document, score) for document, score in self.vector_store.search(vector[0], limit, filters)]
