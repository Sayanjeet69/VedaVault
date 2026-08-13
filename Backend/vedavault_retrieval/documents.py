"""Corpus-independent retrieval document representation and adapters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class RetrievalDocument:
    """A text unit with immutable identifier and serializable metadata."""

    document_id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id must not be empty")
        if not self.text.strip():
            raise ValueError("document text must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {"document_id": self.document_id, "text": self.text, "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RetrievalDocument":
        return cls(str(value["document_id"]), str(value["text"]), value["metadata"])


def deterministic_document_id(passage_id: str, text_layer: str, provenance: Mapping[str, Any], ordinal: int) -> str:
    """Create a stable ID even when two sources have identical text."""
    identity = {
        "passage_id": passage_id,
        "text_layer": text_layer,
        "source_id": provenance.get("source_id", ""),
        "raw_file": provenance.get("raw_file", ""),
        "raw_field": provenance.get("raw_field", ""),
        "ordinal": ordinal,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"{passage_id}:{text_layer}:{digest}"


def _language_for_layer(layer: str, item: Mapping[str, Any]) -> str:
    if layer == "sanskrit":
        return "Sanskrit"
    if layer == "transliteration":
        return "Transliteration"
    return str(item.get("language", "Unknown"))


def documents_from_passages(passages: Iterable[Mapping[str, Any]]) -> list[RetrievalDocument]:
    """Adapt canonical corpus passages without assuming a particular work."""
    documents: list[RetrievalDocument] = []
    for passage in passages:
        for layer in ("sanskrit", "transliteration", "translations", "commentaries"):
            for ordinal, item in enumerate(passage.get(layer, [])):
                text = item.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                provenance = item.get("provenance", {})
                metadata: dict[str, Any] = {
                    "passage_id": passage["passage_id"],
                    "work": passage["work"],
                    "chapter": passage["chapter"],
                    "verse": passage["verse"],
                    "language": _language_for_layer(layer, item),
                    "text_layer": layer,
                    "source": dict(provenance),
                    "provenance": dict(provenance),
                }
                if "translator" in item:
                    metadata["translator"] = item["translator"]
                if "author" in item:
                    metadata["author"] = item["author"]
                documents.append(
                    RetrievalDocument(
                        deterministic_document_id(str(passage["passage_id"]), layer, provenance, ordinal),
                        text,
                        metadata,
                    )
                )
    return documents


def corpus_documents(corpus_path: Path) -> list[RetrievalDocument]:
    """Load a canonical processed corpus and convert its text entries to documents."""
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    passages = corpus.get("passages")
    if not isinstance(passages, list):
        raise ValueError("canonical corpus must contain a 'passages' list")
    return documents_from_passages(passages)
