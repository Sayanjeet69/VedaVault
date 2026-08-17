"""Offline tests for model-independent evidence bundle conversion."""

from __future__ import annotations

import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Backend"))

from vedavault_retrieval import EvidenceBundle, RetrievalDocument, RetrievalResult  # noqa: E402


class EvidenceBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.first_metadata = {
            "passage_id": "BG_02_47",
            "chapter": 2,
            "verse": 47,
            "text_layer": "translations",
            "source": {"source_id": "gita-source", "raw_file": "gita.json"},
            "provenance": {"source_id": "gita-source"},
        }
        self.results = [
            RetrievalResult(RetrievalDocument("first", "Act without attachment.", self.first_metadata), 0.91),
            RetrievalResult(
                RetrievalDocument(
                    "second",
                    "The Self is eternal.",
                    {"passage_id": "BG_02_20", "chapter": 2, "verse": 20, "text_layer": "translations", "source": {"source_id": "other"}},
                ),
                0.73,
            ),
        ]

    def test_bundle_preserves_query_ranking_and_evidence_fields(self) -> None:
        bundle = EvidenceBundle.from_retrieval("How should I act?", self.results)
        self.assertEqual(bundle.query, "How should I act?")
        self.assertEqual([item.document_id for item in bundle.items], ["first", "second"])
        first = bundle.items[0]
        self.assertEqual((first.passage_id, first.chapter, first.verse), ("BG_02_47", 2, 47))
        self.assertEqual((first.text, first.text_layer, first.score), ("Act without attachment.", "translations", 0.91))
        self.assertEqual(first.source["source_id"], "gita-source")
        self.assertEqual(first.metadata["provenance"]["source_id"], "gita-source")

    def test_empty_results_create_a_valid_empty_bundle(self) -> None:
        bundle = EvidenceBundle.from_retrieval("What is dharma?", [])
        self.assertEqual(bundle.items, ())

    def test_conversion_is_immutable_and_does_not_mutate_retrieval_results(self) -> None:
        bundle = EvidenceBundle.from_retrieval(
            "How should I act?", self.results, retrieval_configuration={"text_layers": ["translations"]}
        )
        self.first_metadata["source"]["source_id"] = "changed-after-conversion"
        self.assertEqual(bundle.items[0].source["source_id"], "gita-source")
        self.assertEqual(bundle.retrieval_configuration["text_layers"], ("translations",))
        with self.assertRaises(TypeError):
            bundle.items[0].metadata["chapter"] = 3  # type: ignore[index]
        with self.assertRaises(FrozenInstanceError):
            bundle.query = "different"  # type: ignore[misc]
        self.assertEqual(self.results[0].document.metadata["source"]["source_id"], "changed-after-conversion")

    def test_equivalent_inputs_produce_equivalent_deterministic_bundles(self) -> None:
        first = EvidenceBundle.from_retrieval("How should I act?", self.results)
        second = EvidenceBundle.from_retrieval("How should I act?", list(self.results))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
