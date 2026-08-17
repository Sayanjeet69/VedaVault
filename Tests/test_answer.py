"""Offline tests for the deterministic future-answer contract."""

from __future__ import annotations

import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Backend"))

from vedavault_retrieval import (  # noqa: E402
    ANSWER_CONTRACT_RULES,
    AnswerContract,
    AnswerMode,
    EvidenceBundle,
    GroundingContext,
    RetrievalDocument,
    RetrievalResult,
    ScripturalClaim,
)


class AnswerContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = {
            "passage_id": "BG_02_47",
            "chapter": 2,
            "verse": 47,
            "text_layer": "translations",
            "source": {"source_id": "gita-source"},
        }
        results = [
            RetrievalResult(RetrievalDocument("first", "Act without attachment.", self.metadata), 0.91),
            RetrievalResult(
                RetrievalDocument(
                    "second",
                    "The Self is eternal.",
                    {"passage_id": "BG_02_20", "chapter": 2, "verse": 20, "text_layer": "translations", "source": {"source_id": "other"}},
                ),
                0.73,
            ),
        ]
        self.context = GroundingContext.from_evidence_bundle(EvidenceBundle.from_retrieval("How should I act?", results))
        self.claim = ScripturalClaim("The evidence teaches action without attachment.", ("BG_02_47",))

    def test_textual_mode_preserves_query_and_citations(self) -> None:
        answer = AnswerContract.from_grounding_context(self.context, AnswerMode.TEXTUAL, (self.claim,))
        self.assertEqual((answer.query, answer.mode), ("How should I act?", AnswerMode.TEXTUAL))
        self.assertEqual(answer.cited_verse_ids, ("BG_02_47",))
        self.assertEqual(answer.scriptural_claims[0], self.claim)

    def test_philosophical_and_application_modes_keep_semantic_layers_separate(self) -> None:
        philosophical = AnswerContract.from_grounding_context(
            self.context, AnswerMode.PHILOSOPHICAL, (self.claim,), interpretation="This supports a synthesis about disciplined action."
        )
        application = AnswerContract.from_grounding_context(
            self.context,
            AnswerMode.APPLICATION,
            (self.claim,),
            interpretation="The teaching can be interpreted as emphasizing disciplined action.",
            application="A future answer may relate this principle to the user's situation without calling that application scripture.",
        )
        self.assertIsNone(philosophical.application)
        self.assertEqual(application.scriptural_claims[0].statement, self.claim.statement)
        self.assertIn("user's situation", application.application)

    def test_multiple_citations_and_deterministic_serialization(self) -> None:
        claim = ScripturalClaim("Two passages jointly support this teaching.", ("BG_02_47", "BG_02_20", "BG_02_47"))
        answer = AnswerContract.from_grounding_context(self.context, AnswerMode.TEXTUAL, (claim,))
        self.assertEqual(answer.cited_verse_ids, ("BG_02_47", "BG_02_20"))
        self.assertEqual(answer.to_json(), answer.to_json())
        self.assertIn('"scriptural_teaching"', answer.to_json())
        self.assertIn("canonical verse identifiers", ANSWER_CONTRACT_RULES)

    def test_insufficient_evidence_is_explicit(self) -> None:
        empty_context = GroundingContext.from_evidence_bundle(EvidenceBundle.from_retrieval("What is dharma?", []))
        answer = AnswerContract.from_grounding_context(
            empty_context,
            AnswerMode.TEXTUAL,
            evidence_sufficient=False,
            limitations=("No scriptural evidence was retrieved for this question.",),
        )
        self.assertFalse(answer.evidence_sufficient)
        self.assertEqual(answer.scriptural_claims, ())
        self.assertIn("No scriptural evidence", answer.limitations[0])

    def test_invalid_structures_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AnswerContract("", AnswerMode.TEXTUAL)
        with self.assertRaises(ValueError):
            AnswerContract("query", "textual")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ScripturalClaim("unsupported", ())
        with self.assertRaises(ValueError):
            AnswerContract.from_grounding_context(
                self.context, AnswerMode.TEXTUAL, (ScripturalClaim("unsupported", ("BG_99_99",)),)
            )
        with self.assertRaises(ValueError):
            AnswerContract.from_grounding_context(self.context, AnswerMode.TEXTUAL, (self.claim,), interpretation="not textual")
        with self.assertRaises(ValueError):
            AnswerContract.from_grounding_context(
                self.context, AnswerMode.PHILOSOPHICAL, (self.claim,), application="application in the wrong semantic layer"
            )
        with self.assertRaises(ValueError):
            AnswerContract.from_grounding_context(self.context, AnswerMode.APPLICATION, (self.claim,))
        with self.assertRaises(ValueError):
            AnswerContract.from_grounding_context(self.context, AnswerMode.TEXTUAL, evidence_sufficient=False)
        with self.assertRaises(TypeError):
            ScripturalClaim("not supported", ("BG_02_47",), quotation="fabricated")  # type: ignore[call-arg]

    def test_contract_is_immutable_non_mutating_and_equivalent_for_equivalent_contexts(self) -> None:
        first = AnswerContract.from_grounding_context(self.context, AnswerMode.TEXTUAL, (self.claim,))
        second = AnswerContract.from_grounding_context(self.context, AnswerMode.TEXTUAL, (self.claim,))
        self.metadata["source"]["source_id"] = "changed-after-conversion"
        self.assertEqual(first.evidence_passage_ids, frozenset({"BG_02_47", "BG_02_20"}))
        self.assertEqual(first, second)
        with self.assertRaises(FrozenInstanceError):
            first.query = "different"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
