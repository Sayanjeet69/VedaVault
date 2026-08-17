"""Offline tests for raw-rank-preserving evaluation script orchestration."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Backend"))

from Scripts.evaluate_retrieval import evaluate_question, validate_limits  # noqa: E402
from vedavault_retrieval import EvaluationQuestion, RetrievalDocument, RetrievalResult, evaluate_results  # noqa: E402


def _result(document_id: str, passage_id: str, chapter: int) -> RetrievalResult:
    return RetrievalResult(
        RetrievalDocument(
            document_id,
            document_id,
            {"passage_id": passage_id, "chapter": chapter, "verse": 1, "text_layer": "translation"},
        ),
        1.0,
    )


class EvaluationWorkflowTests(unittest.TestCase):
    def test_raw_mrr_preserves_rank_before_verse_diversification(self) -> None:
        question = EvaluationQuestion(
            "q",
            "question",
            frozenset({"BG_02_20", "BG_03_19"}),
            frozenset({"BG_02_20"}),
            frozenset({"BG_02_20", "BG_03_19"}),
            "test rationale",
        )
        raw_results = [
            _result("a-1", "BG_01_01", 1),
            _result("a-2", "BG_01_01", 1),
            _result("acceptable", "BG_03_19", 3),
            _result("primary", "BG_02_20", 2),
            _result("irrelevant", "BG_04_38", 4),
        ]
        evaluation = evaluate_question(
            question,
            raw_results,
            context_k=5,
            diagnostic_k=10,
            deduplicate_by_verse=True,
        )
        self.assertAlmostEqual(evaluation.raw_at_5.primary_reciprocal_rank, 1 / 4)
        self.assertNotAlmostEqual(evaluation.raw_at_5.primary_reciprocal_rank, 1 / 3)
        self.assertAlmostEqual(evaluation.raw_at_5.acceptable_reciprocal_rank, 1 / 3)
        self.assertEqual([result.document.metadata["passage_id"] for result in evaluation.context_results], ["BG_01_01", "BG_03_19", "BG_02_20", "BG_04_38"])
        self.assertIsNotNone(evaluation.diversified_context)
        self.assertEqual(evaluation.diversified_context.duplicate_result_rate, 0.0)  # type: ignore[union-attr]

    def test_raw_top_ten_diagnostics_are_not_limited_by_context_k(self) -> None:
        question = EvaluationQuestion("q", "question", frozenset({"BG_02_47"}))
        raw_results = [
            _result("a-1", "BG_01_01", 1),
            _result("a-2", "BG_01_01", 1),
            _result("b", "BG_01_02", 1),
            _result("c", "BG_02_01", 2),
            _result("d", "BG_02_02", 2),
            _result("e-1", "BG_03_01", 3),
            _result("e-2", "BG_03_01", 3),
            _result("f", "BG_04_01", 4),
            _result("primary", "BG_02_47", 5),
            _result("g", "BG_05_01", 6),
        ]
        evaluation = evaluate_question(
            question,
            raw_results,
            context_k=5,
            diagnostic_k=10,
            deduplicate_by_verse=False,
        )
        self.assertEqual(len(evaluation.context_results), 5)
        self.assertAlmostEqual(evaluation.raw_at_5.primary_recall_at_k, 0.0)
        self.assertAlmostEqual(evaluation.raw_at_10.primary_recall_at_k, 1.0)
        self.assertEqual(evaluation.raw_at_5.diagnostic_result_count, 10)
        self.assertAlmostEqual(evaluation.raw_at_5.duplicate_result_rate, 0.2)
        self.assertEqual((evaluation.raw_at_5.unique_passage_count, evaluation.raw_at_5.chapter_count), (8, 6))

    def test_raw_metrics_and_diversified_context_diagnostics_are_separate(self) -> None:
        question = EvaluationQuestion("q", "question", frozenset({"BG_02_47"}))
        raw_results = [
            _result("first", "BG_01_01", 1),
            _result("duplicate", "BG_01_01", 1),
            _result("primary", "BG_02_47", 2),
        ]
        evaluation = evaluate_question(
            question,
            raw_results,
            context_k=3,
            diagnostic_k=10,
            deduplicate_by_verse=True,
        )
        self.assertEqual(evaluation.raw_at_5, evaluate_results(question, raw_results, 5, diagnostic_k=10))
        self.assertAlmostEqual(evaluation.raw_at_5.primary_reciprocal_rank, 1 / 3)
        self.assertAlmostEqual(evaluation.raw_at_5.duplicate_result_rate, 1 / 3)
        self.assertEqual(evaluation.diversified_context.context_size, 2)  # type: ignore[union-attr]
        self.assertAlmostEqual(evaluation.diversified_context.duplicate_result_rate, 0.0)  # type: ignore[union-attr]

    def test_no_diversification_keeps_the_existing_raw_evaluation_path(self) -> None:
        question = EvaluationQuestion("q", "question", frozenset({"BG_02_47"}))
        raw_results = [_result("irrelevant", "BG_01_01", 1), _result("primary", "BG_02_47", 2)]
        evaluation = evaluate_question(
            question,
            raw_results,
            context_k=5,
            diagnostic_k=10,
            deduplicate_by_verse=False,
        )
        self.assertEqual(evaluation.raw_at_5, evaluate_results(question, raw_results, 5, diagnostic_k=10))
        self.assertEqual(evaluation.raw_at_10, evaluate_results(question, raw_results, 10, diagnostic_k=10))
        self.assertEqual(evaluation.context_results, tuple(raw_results))
        self.assertIsNone(evaluation.diversified_context)
        with self.assertRaises(ValueError):
            validate_limits(context_k=5, diagnostic_k=10, candidate_limit=9)


if __name__ == "__main__":
    unittest.main()
