"""Unit tests for corpus-agnostic retrieval components; no model download required."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Backend"))
from vedavault_retrieval import (  # noqa: E402
    E5_PROMPT_PROFILE,
    NO_PROMPT_PROFILE,
    EmbeddingConfiguration,
    EmbeddingProvider,
    IndexCompatibilityError,
    IndexManifest,
    IndexManifestError,
    LocalVectorStore,
    RetrievalDocument,
    Retriever,
    SentenceTransformerEmbeddingProvider,
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


class RecordingSentenceTransformer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append(list(texts))
        self.last_kwargs = kwargs
        return np.asarray([[float(len(text)), float(ord(text[-1]))] for text in texts], dtype=np.float32)


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
        self.store.set_manifest(IndexManifest(EmbeddingConfiguration("toy", 3, "none", "", "", None)))
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

    def test_sentence_transformer_batches_in_order(self) -> None:
        model = RecordingSentenceTransformer()
        provider = SentenceTransformerEmbeddingProvider(model=model, batch_size=2)
        vectors = provider.embed(["one", "two", "three", "four", "five"])
        self.assertEqual(model.calls, [["one", "two", "three", "four", "five"]])
        self.assertEqual(vectors.shape, (5, 2))
        self.assertEqual(vectors[:, 1].tolist(), [101.0, 111.0, 101.0, 114.0, 101.0])
        self.assertTrue(model.last_kwargs["normalize_embeddings"])
        self.assertFalse(model.last_kwargs["show_progress_bar"])
        self.assertEqual(model.last_kwargs["batch_size"], 2)

    def test_sentence_transformer_reuses_duplicate_document_embeddings(self) -> None:
        model = RecordingSentenceTransformer()
        provider = SentenceTransformerEmbeddingProvider(model=model, batch_size=2)
        vectors = provider.embed_documents(["same", "other", "same"])
        self.assertEqual(model.calls, [["passage: same", "passage: other"]])
        self.assertTrue(np.array_equal(vectors[0], vectors[2]))
        workload = provider.document_workload(["same", "other", "same"])
        self.assertEqual((workload.input_count, workload.unique_input_count, workload.batch_count), (3, 2, 1))

    def test_sentence_transformer_e5_roles_and_environment_configuration(self) -> None:
        original = {name: os.environ.get(name) for name in ("VEDAVAULT_EMBEDDING_MODEL", "VEDAVAULT_EMBEDDING_BATCH_SIZE", "VEDAVAULT_EMBEDDING_CPU_THREADS", "VEDAVAULT_EMBEDDING_PROMPT_PROFILE")}
        self.addCleanup(lambda: [os.environ.pop(name, None) if value is None else os.environ.__setitem__(name, value) for name, value in original.items()])
        os.environ.update({"VEDAVAULT_EMBEDDING_MODEL": "example/model", "VEDAVAULT_EMBEDDING_BATCH_SIZE": "3", "VEDAVAULT_EMBEDDING_CPU_THREADS": "2"})
        model = RecordingSentenceTransformer()
        provider = SentenceTransformerEmbeddingProvider(model=model, prompt_profile=E5_PROMPT_PROFILE)
        self.assertEqual((provider.model_name, provider.batch_size, provider.cpu_threads), ("example/model", 3, 2))
        provider.embed_documents(["a", "b"])
        provider.embed_query("question")
        self.assertEqual(model.calls, [["passage: a", "passage: b"], ["query: question"]])

    def test_sentence_transformer_non_e5_profile_has_no_e5_prefixes(self) -> None:
        model = RecordingSentenceTransformer()
        provider = SentenceTransformerEmbeddingProvider(
            model_name="sentence-transformers/LaBSE", model=model, prompt_profile=NO_PROMPT_PROFILE
        )
        provider.embed_documents(["a"])
        provider.embed_query("question")
        self.assertEqual(model.calls, [["a"], ["question"]])

    def test_sentence_transformer_non_e5_profile_can_be_selected_by_environment(self) -> None:
        original = os.environ.get("VEDAVAULT_EMBEDDING_PROMPT_PROFILE")
        self.addCleanup(
            lambda: os.environ.pop("VEDAVAULT_EMBEDDING_PROMPT_PROFILE", None)
            if original is None else os.environ.__setitem__("VEDAVAULT_EMBEDDING_PROMPT_PROFILE", original)
        )
        os.environ["VEDAVAULT_EMBEDDING_PROMPT_PROFILE"] = "none"
        model = RecordingSentenceTransformer()
        provider = SentenceTransformerEmbeddingProvider(model_name="sentence-transformers/LaBSE", model=model)
        provider.embed_documents(["a"])
        self.assertEqual(model.calls, [["a"]])

    def test_embedding_configuration_is_deterministic_and_semantic(self) -> None:
        provider = SentenceTransformerEmbeddingProvider(model=RecordingSentenceTransformer(), max_seq_length=256)
        self.assertEqual(provider.index_configuration(2), provider.index_configuration(2))
        self.assertEqual(
            provider.index_configuration(2).to_dict()["document_prefix"], "passage: "
        )

    def test_manifest_round_trip_and_compatible_index_loading(self) -> None:
        provider = SentenceTransformerEmbeddingProvider(model=RecordingSentenceTransformer())
        vectors = provider.embed_documents([document.text for document in self.documents])
        store = LocalVectorStore()
        store.add(self.documents, vectors)
        store.set_manifest(IndexManifest(provider.index_configuration(vectors.shape[1])))
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(directory))
        store.save(directory)
        restored = LocalVectorStore.load(directory, embedding_provider=provider)
        self.assertEqual(restored.manifest.embedding, provider.index_configuration(2))
        self.assertTrue((directory / "index_manifest.json").is_file())
        self.assertEqual(len(Retriever(provider, restored).retrieve("karma", limit=1)), 1)

    def test_index_rejects_incompatible_model_or_prompt_configuration(self) -> None:
        provider = SentenceTransformerEmbeddingProvider(model=RecordingSentenceTransformer())
        vectors = provider.embed_documents([document.text for document in self.documents])
        store = LocalVectorStore()
        store.add(self.documents, vectors)
        store.set_manifest(IndexManifest(provider.index_configuration(vectors.shape[1])))
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(directory))
        store.save(directory)
        incompatible = SentenceTransformerEmbeddingProvider(
            model_name="another/model", model=RecordingSentenceTransformer(), prompt_profile=NO_PROMPT_PROFILE
        )
        with self.assertRaises(IndexCompatibilityError):
            LocalVectorStore.load(directory, embedding_provider=incompatible)

    def test_index_rejects_incompatible_embedding_dimension(self) -> None:
        provider = SentenceTransformerEmbeddingProvider(model=RecordingSentenceTransformer())
        vectors = provider.embed_documents([document.text for document in self.documents])
        store = LocalVectorStore()
        store.add(self.documents, vectors)
        store.set_manifest(IndexManifest(provider.index_configuration(vectors.shape[1])))
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(directory))
        store.save(directory)
        manifest_path = directory / "index_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["embedding"]["embedding_dimension"] = 3
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(IndexManifestError):
            LocalVectorStore.load(directory)

    def test_index_rejects_missing_or_invalid_manifest(self) -> None:
        provider = SentenceTransformerEmbeddingProvider(model=RecordingSentenceTransformer())
        vectors = provider.embed_documents([document.text for document in self.documents])
        store = LocalVectorStore()
        store.add(self.documents, vectors)
        store.set_manifest(IndexManifest(provider.index_configuration(vectors.shape[1])))
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(directory))
        store.save(directory)
        manifest_path = directory / "index_manifest.json"
        manifest_path.unlink()
        with self.assertRaises(IndexManifestError):
            LocalVectorStore.load(directory)
        manifest_path.write_text("not json", encoding="utf-8")
        with self.assertRaises(IndexManifestError):
            LocalVectorStore.load(directory)

    def test_sentence_transformer_max_sequence_length_configuration(self) -> None:
        model = RecordingSentenceTransformer()
        model.max_seq_length = 512
        provider = SentenceTransformerEmbeddingProvider(model=model, max_seq_length=256)
        provider.embed(["text"])
        self.assertEqual(model.max_seq_length, 256)

    def test_sentence_transformer_rejects_invalid_batch_configuration(self) -> None:
        with self.assertRaises(ValueError):
            SentenceTransformerEmbeddingProvider(model=RecordingSentenceTransformer(), batch_size=0)

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
