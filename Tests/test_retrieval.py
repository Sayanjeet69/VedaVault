"""Unit tests for corpus-agnostic retrieval components; no model download required."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Backend"))
from vedavault_retrieval import (  # noqa: E402
    EmbeddingProvider,
    LocalVectorStore,
    RetrievalDocument,
    Retriever,
    WordChunker,
    corpus_documents,
    deterministic_document_id,
)


class ToyEmbeddingProvider(EmbeddingProvider):
    """Stable small vectors that make retrieval behavior testable offline."""

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float(any(word in lowered for word in ("karma", "action", "fruits"))),
                    float(any(word in lowered for word in ("self", "soul", "atman"))),
                    1.0,
                ]
            )
        return np.asarray(vectors, dtype=np.float32)


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = ToyEmbeddingProvider()
        self.documents = [
            RetrievalDocument("BG_02_47:translation:a", "Act without attachment to the fruits of action.", {"passage_id": "BG_02_47", "work": "Example", "chapter": 2, "verse": 47, "language": "English", "text_layer": "translations", "source": {"source_id": "one"}}),
            RetrievalDocument("BG_02_20:translation:b", "The Self is eternal and cannot be destroyed.", {"passage_id": "BG_02_20", "work": "Example", "chapter": 2, "verse": 20, "language": "English", "text_layer": "translations", "source": {"source_id": "two"}}),
        ]
        self.store = LocalVectorStore()
        self.store.add(self.documents, self.provider.embed([document.text for document in self.documents]))

    def test_document_creation_and_deterministic_ids(self) -> None:
        provenance = {"source_id": "source", "raw_file": "file", "raw_field": "field"}
        self.assertEqual(
            deterministic_document_id("PX_01", "translations", provenance, 2),
            deterministic_document_id("PX_01", "translations", provenance, 2),
        )
        with self.assertRaises(ValueError):
            RetrievalDocument("", "text")

    def test_chunking_preserves_parent_metadata(self) -> None:
        document = RetrievalDocument("doc", "one two three four five six", {"work": "Example"})
        chunks = WordChunker(max_words=3, overlap_words=1).chunk(document)
        self.assertEqual([chunk.document_id for chunk in chunks], ["doc:chunk:0000", "doc:chunk:0001", "doc:chunk:0002"])
        self.assertEqual(chunks[1].metadata["parent_document_id"], "doc")
        self.assertEqual(chunks[1].metadata["work"], "Example")

    def test_embedding_index_construction_and_metadata_preservation(self) -> None:
        self.assertEqual(self.store.size, 2)
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(directory))
        self.store.save(directory)
        restored = LocalVectorStore.load(directory)
        result = restored.search(self.provider.embed(["karma action"])[0], 1)[0]
        self.assertEqual(result[0].metadata["source"]["source_id"], "one")

    def test_retrieval_and_metadata_filters(self) -> None:
        retriever = Retriever(self.provider, self.store)
        results = retriever.retrieve("karma yoga action", limit=1, filters={"chapter": 2, "text_layer": "translations"})
        self.assertEqual(results[0].document.metadata["passage_id"], "BG_02_47")
        self.assertGreater(results[0].score, 0)
        self.assertEqual(retriever.retrieve("karma", filters={"language": "Hindi"}), [])

    def test_empty_invalid_queries(self) -> None:
        retriever = Retriever(self.provider, self.store)
        for query in ("", "   ", None):
            with self.assertRaises(ValueError):
                retriever.retrieve(query)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            retriever.retrieve("valid", limit=0)

    def test_canonical_corpus_documents_keep_required_metadata(self) -> None:
        corpus_path = ROOT / "Data" / "Processed" / "Bhagavad_Gita" / "corpus.json"
        documents = corpus_documents(corpus_path)
        self.assertGreater(len(documents), 701)
        first = documents[0]
        self.assertEqual(first.metadata["passage_id"], "BG_01_01")
        self.assertEqual(first.metadata["work"], "Bhagavad Gita")
        self.assertIn("source_id", first.metadata["provenance"])


if __name__ == "__main__":
    unittest.main()
