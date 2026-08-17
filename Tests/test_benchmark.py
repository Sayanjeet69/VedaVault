"""Offline tests for the versioned multilingual retrieval benchmark and scorer."""

from __future__ import annotations

from collections import Counter
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Backend"))

from vedavault_retrieval import (  # noqa: E402
    BENCHMARK_VERSION,
    BREAKDOWN_DIMENSIONS,
    CATEGORIES,
    DIFFICULTY_TAGS,
    EvaluationQuestion,
    RetrievalDocument,
    RetrievalResult,
    SupportedLanguage,
    WritingScript,
    aggregate_evaluations,
    evaluate_results,
    load_retrieval_benchmark,
    validate_benchmark_corpus_references,
)


BENCHMARK_PATH = ROOT / "Evaluation" / "bhagavad_gita_retrieval.json"
CORPUS_PATH = ROOT / "Data" / "Processed" / "Bhagavad_Gita" / "corpus.json"


def _result(document_id: str, verse_id: str, chapter: int) -> RetrievalResult:
    return RetrievalResult(RetrievalDocument(document_id, document_id, {"passage_id": verse_id, "chapter": chapter}), 1.0)


class MultilingualBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmark = load_retrieval_benchmark(BENCHMARK_PATH)

    def test_loading_is_deterministic_versioned_and_in_target_size_range(self) -> None:
        reloaded = load_retrieval_benchmark(BENCHMARK_PATH)
        self.assertEqual(self.benchmark, reloaded)
        self.assertEqual(self.benchmark.benchmark_version, BENCHMARK_VERSION)
        self.assertEqual(self.benchmark.work, "Bhagavad Gita")
        self.assertGreaterEqual(len(self.benchmark.questions), 48)
        self.assertLessEqual(len(self.benchmark.questions), 56)
        self.assertEqual(self.benchmark.to_dict(), reloaded.to_dict())

    def test_schema_ground_truth_and_corpus_references_are_valid(self) -> None:
        validate_benchmark_corpus_references(self.benchmark, CORPUS_PATH)
        question_ids = [question.question_id for question in self.benchmark.questions]
        self.assertEqual(len(question_ids), len(set(question_ids)))
        for question in self.benchmark.questions:
            with self.subTest(question_id=question.question_id):
                self.assertTrue(question.question.strip())
                self.assertIn(question.category, CATEGORIES)
                self.assertIsInstance(question.input_language, SupportedLanguage)
                self.assertIsInstance(question.writing_script, WritingScript)
                self.assertTrue(question.primary_passage_ids)
                self.assertTrue(question.acceptable_passage_ids)
                self.assertTrue(question.primary_passage_ids <= question.acceptable_passage_ids)
                self.assertTrue(question.difficulty_tags <= DIFFICULTY_TAGS)
                self.assertTrue(question.rationale.strip())

    def test_metadata_and_language_coverage_are_meaningful(self) -> None:
        questions = self.benchmark.questions
        language_counts = Counter(question.input_language.value for question in questions)
        for language in ("en", "hi", "bn", "sa"):
            self.assertGreaterEqual(language_counts[language], 5)
        for language in ("ta", "te", "mr", "gu"):
            self.assertGreaterEqual(language_counts[language], 3)
        self.assertGreaterEqual(sum(question.transliterated for question in questions), 4)
        self.assertGreaterEqual(sum(question.code_switched for question in questions), 3)
        self.assertGreaterEqual(sum(question.imperfect_input for question in questions), 3)
        self.assertGreater(sum(question.cross_language for question in questions), 0)
        self.assertTrue(
            any(question.writing_script is WritingScript.LATIN and question.input_language is not SupportedLanguage.ENGLISH for question in questions)
        )
        self.assertTrue(any(question.code_switched and question.transliterated for question in questions))
        self.assertTrue(any(len(question.difficulty_tags) > 1 for question in questions))

    def test_loader_rejects_representative_malformed_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.json"
            malformed_items = (
                ("invalid_id", "id", "invalid id"),
                ("non_boolean", "code_switched", "false"),
                ("duplicate_verse", "acceptable_verses", ["BG_02_47", "BG_02_47"]),
                ("primary_not_acceptable", "primary_verses", ["BG_01_01"]),
            )
            for name, field, value in malformed_items:
                with self.subTest(name=name):
                    benchmark_data = self.benchmark.to_dict()
                    benchmark_data["items"][0][field] = value
                    path.write_text(json.dumps(benchmark_data), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_retrieval_benchmark(path)

    def test_primary_and_acceptable_metrics_preserve_rank_and_duplicate_diagnostics(self) -> None:
        question = EvaluationQuestion(
            "q",
            "question",
            frozenset({"BG_02_47", "BG_02_48", "BG_03_19"}),
            frozenset({"BG_02_47", "BG_02_48"}),
            frozenset({"BG_02_47", "BG_02_48", "BG_03_19"}),
            "test rationale",
            "action-and-results",
            SupportedLanguage.ENGLISH,
            WritingScript.LATIN,
            difficulty_tags=frozenset({"semantic"}),
        )
        results = [
            _result("one", "BG_03_19", 3),
            _result("two", "BG_02_47", 2),
            _result("three", "BG_02_47", 2),
            _result("four", "BG_04_38", 4),
            _result("five", "BG_02_48", 2),
            _result("six", "BG_05_26", 5),
        ]
        metrics = evaluate_results(question, results, 5, diagnostic_k=10)
        self.assertAlmostEqual(metrics.primary_recall_at_k, 1.0)
        self.assertAlmostEqual(metrics.acceptable_recall_at_k, 1.0)
        self.assertAlmostEqual(metrics.acceptable_precision_at_k, 3 / 5)
        self.assertAlmostEqual(metrics.primary_reciprocal_rank, 1 / 2)
        self.assertAlmostEqual(metrics.acceptable_reciprocal_rank, 1.0)
        self.assertAlmostEqual(metrics.duplicate_result_rate, 1 / 6)
        self.assertEqual((metrics.unique_passage_count, metrics.repeated_passage_count, metrics.chapter_count), (5, 1, 4))
        self.assertEqual(metrics.primary_found, frozenset({"BG_02_47", "BG_02_48"}))
        self.assertEqual(metrics.acceptable_only_found, frozenset({"BG_03_19"}))
        self.assertEqual(metrics.expected_found, metrics.acceptable_found)
        recall_at_ten = evaluate_results(question, results, 10)
        self.assertAlmostEqual(recall_at_ten.primary_recall_at_k, 1.0)

    def test_precision_uses_requested_cutoff_when_fewer_results_exist(self) -> None:
        question = EvaluationQuestion("q", "question", frozenset({"BG_02_47"}))
        metrics = evaluate_results(question, [_result("one", "BG_02_47", 2)], 5)
        self.assertAlmostEqual(metrics.acceptable_precision_at_k, 1 / 5)
        self.assertAlmostEqual(metrics.primary_reciprocal_rank, 1.0)

    def test_aggregation_supports_every_declared_metadata_breakdown(self) -> None:
        metric = evaluate_results(
            self.benchmark.questions[0],
            [_result("one", "BG_02_47", 2)],
            5,
        )
        scored = [(question, metric) for question in self.benchmark.questions]
        for dimension in BREAKDOWN_DIMENSIONS:
            with self.subTest(dimension=dimension):
                groups = aggregate_evaluations(scored, by=dimension)
                self.assertTrue(groups)
                if dimension == "difficulty":
                    self.assertGreaterEqual(sum(group.sample_count for group in groups.values()), len(scored))
                else:
                    self.assertEqual(sum(group.sample_count for group in groups.values()), len(scored))
        language_groups = aggregate_evaluations(scored, by="input_language")
        self.assertEqual(language_groups["en"].sample_count, 10)
        self.assertEqual(language_groups["bn"].sample_count, 7)


if __name__ == "__main__":
    unittest.main()
