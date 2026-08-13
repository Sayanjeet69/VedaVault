"""Integration tests for the Bhagavad Gita normalization layer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
from ingest_bhagavad_gita import (  # noqa: E402
    CONTAMINATED_FIELDS,
    SOURCE_BHAGWAD_CSV,
    SOURCE_GEETA_CSV,
    build_corpus,
)


class BhagavadGitaIngestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = build_corpus()
        cls.passages = cls.corpus["passages"]

    def test_has_all_701_unique_canonical_verse_ids(self) -> None:
        ids = [passage["passage_id"] for passage in self.passages]
        self.assertEqual(len(ids), 701)
        self.assertEqual(len(set(ids)), 701)
        self.assertEqual(ids[0], "BG_01_01")
        self.assertEqual(ids[-1], "BG_18_78")

    def test_covers_chapters_one_through_eighteen(self) -> None:
        self.assertEqual({passage["chapter"] for passage in self.passages}, set(range(1, 19)))

    def test_all_three_sources_align_on_701_verse_ids(self) -> None:
        self.assertEqual(
            self.corpus["source_alignment"]["source_record_counts"],
            {
                "dharmicdata_json": 701,
                "bhagwad_gita_csv": 701,
                "geeta_dataset_csv": 701,
            },
        )
        for passage in self.passages:
            self.assertEqual(
                {item["source_id"] for item in passage["provenance"]},
                {"dharmicdata_json", "bhagwad_gita_csv", "geeta_dataset_csv"},
            )

    def test_preserves_multiple_translations_and_commentaries(self) -> None:
        first = self.passages[0]
        self.assertGreaterEqual(len(first["translations"]), 10)
        self.assertGreaterEqual(len(first["commentaries"]), 10)
        translators = {item["translator"] for item in first["translations"]}
        self.assertIn("swami sivananda", translators)
        self.assertIn("Bhagwad_Gita.csv", translators)
        self.assertIn("geeta_dataset.csv", translators)
        self.assertTrue(all("provenance" in item for item in first["translations"] + first["commentaries"]))

    def test_known_contaminated_hindi_fields_are_flagged_with_raw_values(self) -> None:
        expected = {
            (source_id, chapter, verse, field)
            for source_id, chapter, verse, field in CONTAMINATED_FIELDS
        }
        found = set()
        for passage in self.passages:
            for flag in passage["flags"]:
                provenance = flag["provenance"]
                if flag["code"] == "contaminated_language_field":
                    found.add(
                        (
                            provenance["source_id"],
                            passage["chapter"],
                            passage["verse"],
                            provenance["raw_field"],
                        )
                    )
                    self.assertTrue(flag["raw_value"])
        self.assertEqual(found, expected)

    def test_semantic_missing_value_sentinels_are_flagged_not_ingested_as_content(self) -> None:
        sentinel_flags = [
            flag for passage in self.passages for flag in passage["flags"] if flag["code"] == "semantic_missing_value"
        ]
        self.assertGreater(len(sentinel_flags), 0)
        self.assertTrue(all("no commentary" in flag["raw_value"].lower() or "no translation" in flag["raw_value"].lower() or "did not comment" in flag["raw_value"].lower() for flag in sentinel_flags))
        content = [
            item["text"].lower()
            for passage in self.passages
            for field in ("translations", "commentaries")
            for item in passage[field]
        ]
        self.assertFalse(any("no commentary" in value or "no translation" in value for value in content))


if __name__ == "__main__":
    unittest.main()
