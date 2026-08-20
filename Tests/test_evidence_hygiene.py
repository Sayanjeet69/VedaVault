"""Offline coverage for provenance-aware grounding evidence substitution."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Backend"))

from vedavault_retrieval import (  # noqa: E402
    AnswerContract,
    AnswerMode,
    EvidenceBundle,
    EvidenceHygieneError,
    EvidenceHygienePolicy,
    GroundingContext,
    RetrievalDocument,
    RetrievalResult,
    ScripturalClaim,
)


CONTAMINATED = (
    ("bhagwad_gita_csv", "WordMeaning", "English"),
    (
        "dharmicdata_json",
        "translations.swami adidevananda",
        "English",
    ),
    (
        "dharmicdata_json",
        "translations.swami gambirananda",
        "English",
    ),
    (
        "dharmicdata_json",
        "translations.sri harikrishnadas goenka",
        "Hindi",
    ),
)


def _translation(
    text: str, language: str, source_id: str, raw_field: str, translator: str
) -> dict[str, object]:
    return {
        "text": text,
        "language": language,
        "translator": translator,
        "provenance": {
            "source_id": source_id,
            "raw_file": "fixture.json",
            "raw_field": raw_field,
            "raw_record_id": "2:47",
        },
    }


def _policy() -> EvidenceHygienePolicy:
    return EvidenceHygienePolicy(
        [
            {
                "passage_id": "BG_02_47",
                "translations": [
                    _translation(
                        "Clean English primary.",
                        "English",
                        "dharmicdata_json",
                        "translations.swami sivananda",
                        "swami sivananda",
                    ),
                    _translation(
                        "Clean English fallback.",
                        "English",
                        "geeta_dataset_csv",
                        "english",
                        "geeta_dataset.csv",
                    ),
                    _translation(
                        "Clean Hindi primary.",
                        "Hindi",
                        "dharmicdata_json",
                        "translations.swami tejomayananda",
                        "swami tejomayananda",
                    ),
                ],
            }
        ]
    )


def _result(
    document_id: str,
    text: str,
    source_id: str,
    raw_field: str,
    language: str,
    score: float = 0.91,
) -> RetrievalResult:
    provenance = {
        "source_id": source_id,
        "raw_file": "selected.json",
        "raw_field": raw_field,
        "raw_record_id": "2:47",
    }
    return RetrievalResult(
        RetrievalDocument(
            document_id,
            text,
            {
                "passage_id": "BG_02_47",
                "chapter": 2,
                "verse": 47,
                "language": language,
                "text_layer": "translations",
                "source": provenance,
                "provenance": provenance,
            },
        ),
        score,
    )


class EvidenceHygieneTests(unittest.TestCase):
    def test_clean_selected_hit_remains_unchanged_and_backward_compatible(self) -> None:
        selected = _result(
            "clean-selected",
            "Already clean.",
            "dharmicdata_json",
            "translations.swami sivananda",
            "English",
        )
        bundle = EvidenceBundle.from_retrieval("question", [selected])
        context = GroundingContext.from_evidence_bundle(bundle, _policy())

        self.assertIs(context.evidence_items[0], bundle.items[0])
        self.assertEqual(context.evidence_items[0].text, "Already clean.")
        self.assertEqual(context.evidence_items[0].source, bundle.items[0].source)
        self.assertIn("document_id: clean-selected", context.to_prompt_context())
        self.assertIn("retrieved_text:\nAlready clean.", context.to_prompt_context())

    def test_all_confirmed_contaminated_groups_are_replaced(self) -> None:
        for source_id, raw_field, language in CONTAMINATED:
            with self.subTest(raw_field=raw_field):
                contaminated_text = f"CONTAMINATED {raw_field}"
                bundle = EvidenceBundle.from_retrieval(
                    "question",
                    [_result("selected-doc", contaminated_text, source_id, raw_field, language)],
                )
                context = GroundingContext.from_evidence_bundle(bundle, _policy())
                grounded = context.evidence_items[0]

                expected = (
                    "Clean Hindi primary." if language == "Hindi" else "Clean English primary."
                )
                self.assertEqual(grounded.text, expected)
                self.assertNotIn(contaminated_text, context.to_prompt_context())
                self.assertEqual(grounded.passage_id, "BG_02_47")
                self.assertEqual(grounded.score, 0.91)
                self.assertEqual(grounded.document_id, "selected-doc")
                self.assertEqual(
                    grounded.metadata["selection_provenance"]["provenance"]["raw_field"],
                    raw_field,
                )
                self.assertEqual(
                    grounded.metadata["grounding_provenance"]["raw_field"],
                    "translations.swami tejomayananda"
                    if language == "Hindi"
                    else "translations.swami sivananda",
                )
                prompt = context.to_prompt_context()
                self.assertIn("selection_document_id: selected-doc", prompt)
                self.assertIn("selection_provenance:", prompt)
                self.assertIn("grounding_provenance:", prompt)
                self.assertIn("grounding_text:", prompt)

    def test_rank_order_scores_and_canonical_ids_are_preserved(self) -> None:
        contaminated = _result(
            "first", "CONTAMINATED", "bhagwad_gita_csv", "WordMeaning", "English", 0.99
        )
        clean = _result(
            "second",
            "Already clean.",
            "geeta_dataset_csv",
            "english",
            "English",
            0.72,
        )
        bundle = EvidenceBundle.from_retrieval("question", [contaminated, clean])
        context = GroundingContext.from_evidence_bundle(bundle, _policy())

        self.assertEqual(
            [(item.document_id, item.passage_id, item.score) for item in context.evidence_items],
            [("first", "BG_02_47", 0.99), ("second", "BG_02_47", 0.72)],
        )
        self.assertEqual(bundle.items[0].text, "CONTAMINATED")
        self.assertEqual(
            bundle.items[0].source["raw_field"],
            "WordMeaning",
        )

    def test_missing_clean_replacement_fails_closed(self) -> None:
        bundle = EvidenceBundle.from_retrieval(
            "question",
            [
                _result(
                    "selected-doc",
                    "CONTAMINATED",
                    "bhagwad_gita_csv",
                    "WordMeaning",
                    "English",
                )
            ],
        )
        with self.assertRaisesRegex(EvidenceHygieneError, "grounding refused"):
            GroundingContext.from_evidence_bundle(bundle, EvidenceHygienePolicy([]))

    def test_answer_contract_and_citation_validation_accept_sanitized_context(self) -> None:
        bundle = EvidenceBundle.from_retrieval(
            "question",
            [
                _result(
                    "selected-doc",
                    "CONTAMINATED",
                    "dharmicdata_json",
                    "translations.swami gambirananda",
                    "English",
                )
            ],
        )
        context = GroundingContext.from_evidence_bundle(bundle, _policy())
        answer = AnswerContract.from_grounding_context(
            context,
            AnswerMode.TEXTUAL,
            (ScripturalClaim("A grounded claim.", ("BG_02_47",)),),
        )

        self.assertEqual(answer.evidence_passage_ids, frozenset({"BG_02_47"}))
        self.assertEqual(answer.scriptural_claims[0].cited_verse_ids, ("BG_02_47",))


if __name__ == "__main__":
    unittest.main()
