"""Retrieval orchestration over VedaVault-controlled embedding and index contracts."""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass

from .embeddings import EmbeddingProvider
from .filters import MetadataFilter
from .vector_store import VectorStore
from .documents import RetrievalDocument


@dataclass(frozen=True)
class RetrievalResult:
    document: RetrievalDocument
    score: float


def deduplicate_by_passage(results: Iterable[RetrievalResult]) -> list[RetrievalResult]:
    """Keep the highest-ranked result per canonical verse, preserving rank order."""
    unique_results: list[RetrievalResult] = []
    seen: set[str] = set()
    for result in results:
        passage_id = result.document.metadata.get("passage_id")
        if not isinstance(passage_id, str) or not passage_id:
            raise ValueError("retrieval result is missing metadata.passage_id")
        if passage_id not in seen:
            seen.add(passage_id)
            unique_results.append(result)
    return unique_results


class Retriever:
    def __init__(self, embedding_provider: EmbeddingProvider, vector_store: VectorStore) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        validate_provider = getattr(vector_store, "validate_embedding_provider", None)
        if getattr(vector_store, "manifest", None) is not None and validate_provider is not None:
            validate_provider(embedding_provider)

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        filters: MetadataFilter | None = None,
        deduplicate_by_verse: bool = False,
        diversity_candidate_limit: int | None = None,
        text_layers: Collection[str] | None = None,
    ) -> list[RetrievalResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if limit < 1:
            raise ValueError("limit must be positive")
        if diversity_candidate_limit is not None and diversity_candidate_limit < limit:
            raise ValueError("diversity_candidate_limit must be at least limit")
        effective_filters = dict(filters or {})
        if text_layers is not None:
            if "text_layer" in effective_filters:
                raise ValueError("text_layers cannot be combined with filters['text_layer']")
            if not text_layers or not all(isinstance(layer, str) and layer for layer in text_layers):
                raise ValueError("text_layers must contain at least one non-empty layer name")
            effective_filters["text_layer"] = tuple(text_layers)
        vector = self.embedding_provider.embed_query(query)
        if vector.ndim != 2 or vector.shape[0] != 1:
            raise ValueError("embedding provider must return one vector for a query")
        candidate_limit = limit
        if deduplicate_by_verse:
            candidate_limit = diversity_candidate_limit or limit * 5
        results = [
            RetrievalResult(document, score)
            for document, score in self.vector_store.search(vector[0], candidate_limit, effective_filters)
        ]
        return deduplicate_by_passage(results)[:limit] if deduplicate_by_verse else results
