"""Document chunking policies independent of source corpus and vector store."""

from __future__ import annotations

from dataclasses import dataclass

from .documents import RetrievalDocument


@dataclass(frozen=True)
class WordChunker:
    """Split documents into overlapping word windows while retaining metadata."""

    max_words: int = 220
    overlap_words: int = 30

    def __post_init__(self) -> None:
        if self.max_words < 1:
            raise ValueError("max_words must be positive")
        if not 0 <= self.overlap_words < self.max_words:
            raise ValueError("overlap_words must be non-negative and less than max_words")

    def chunk(self, document: RetrievalDocument) -> list[RetrievalDocument]:
        words = document.text.split()
        if len(words) <= self.max_words:
            return [document]
        chunks: list[RetrievalDocument] = []
        step = self.max_words - self.overlap_words
        for index, start in enumerate(range(0, len(words), step)):
            text = " ".join(words[start : start + self.max_words])
            if not text:
                break
            metadata = dict(document.metadata)
            metadata.update({"parent_document_id": document.document_id, "chunk_index": index})
            chunks.append(RetrievalDocument(f"{document.document_id}:chunk:{index:04d}", text, metadata))
            if start + self.max_words >= len(words):
                break
        return chunks

    def chunk_all(self, documents: list[RetrievalDocument]) -> list[RetrievalDocument]:
        return [chunk for document in documents for chunk in self.chunk(document)]
