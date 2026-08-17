"""Model-independent evidence transport between retrieval and future grounding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .retrieval import RetrievalResult


def _freeze(value: Any) -> Any:
    """Copy nested metadata into immutable containers for stable evidence."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(nested_value) for key, nested_value in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class EvidenceItem:
    """A single ranked retrieval result with immutable, traceable provenance."""

    document_id: str
    passage_id: str | None
    chapter: int | None
    verse: int | None
    text: str
    text_layer: str | None
    score: float
    source: Mapping[str, Any] | None
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        if self.source is not None:
            object.__setattr__(self, "source", _freeze(self.source))

    @classmethod
    def from_retrieval_result(cls, result: RetrievalResult) -> "EvidenceItem":
        metadata = _freeze(result.document.metadata)
        source = metadata.get("source")
        return cls(
            document_id=result.document.document_id,
            passage_id=_optional_string(metadata.get("passage_id")),
            chapter=_optional_int(metadata.get("chapter")),
            verse=_optional_int(metadata.get("verse")),
            text=result.document.text,
            text_layer=_optional_string(metadata.get("text_layer")),
            score=result.score,
            source=source if isinstance(source, Mapping) else None,
            metadata=metadata,
        )


@dataclass(frozen=True)
class EvidenceBundle:
    """Ordered retrieval evidence for a query, independent of any LLM provider."""

    query: str
    items: tuple[EvidenceItem, ...]
    retrieval_configuration: Mapping[str, Any] | None = field(default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query must be a non-empty string")
        object.__setattr__(self, "items", tuple(self.items))
        if self.retrieval_configuration is not None:
            object.__setattr__(self, "retrieval_configuration", _freeze(self.retrieval_configuration))

    @classmethod
    def from_retrieval(
        cls,
        query: str,
        results: Sequence[RetrievalResult],
        retrieval_configuration: Mapping[str, Any] | None = None,
    ) -> "EvidenceBundle":
        """Convert ranked retrieval results without reordering or mutating them."""
        return cls(
            query=query,
            items=tuple(EvidenceItem.from_retrieval_result(result) for result in results),
            retrieval_configuration=retrieval_configuration,
        )


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None
