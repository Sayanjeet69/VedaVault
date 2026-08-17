"""Offline-friendly retrieval evaluation metrics and diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path

from .retrieval import RetrievalResult, deduplicate_by_passage


@dataclass(frozen=True)
class EvaluationQuestion:
    """A question with primary and acceptable canonical verse judgments."""

    question_id: str
    question: str
    expected_passage_ids: frozenset[str]
    primary_passage_ids: frozenset[str] = frozenset()
    acceptable_passage_ids: frozenset[str] = frozenset()
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.question_id or not self.question.strip() or not self.expected_passage_ids:
            raise ValueError("evaluation questions require an id, question, and expected passage IDs")
        if not self.primary_passage_ids <= self.expected_passage_ids:
            raise ValueError("primary passage IDs must be included in expected passage IDs")
        if not self.acceptable_passage_ids <= self.expected_passage_ids:
            raise ValueError("acceptable passage IDs must be included in expected passage IDs")


@dataclass(frozen=True)
class RetrievalEvaluation:
    """Verse-level metrics plus diagnostics for one retrieval result list."""

    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    duplicate_result_rate: float
    unique_passage_count: int
    repeated_passage_count: int
    chapter_count: int
    expected_found: frozenset[str]


def load_evaluation_questions(path: Path) -> list[EvaluationQuestion]:
    """Load the version-controlled evaluation set without coupling it to a corpus adapter."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("evaluation dataset must be a JSON list")
    questions: list[EvaluationQuestion] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("each evaluation item must be a JSON object")
        required = {"id", "question", "primary_passage_ids", "acceptable_passage_ids", "rationale"}
        if set(item) != required:
            raise ValueError("each evaluation item must contain primary/acceptable passage IDs and rationale")
        primary, acceptable = item["primary_passage_ids"], item["acceptable_passage_ids"]
        if not all(
            isinstance(values, list) and all(isinstance(passage, str) and passage for passage in values)
            for values in (primary, acceptable)
        ):
            raise ValueError("primary_passage_ids and acceptable_passage_ids must be lists of non-empty strings")
        if not isinstance(item["rationale"], str) or not item["rationale"].strip():
            raise ValueError("evaluation rationale must be a non-empty string")
        expected = frozenset(primary) | frozenset(acceptable)
        questions.append(
            EvaluationQuestion(item["id"], item["question"], expected, frozenset(primary), frozenset(acceptable), item["rationale"])
        )
    if len({question.question_id for question in questions}) != len(questions):
        raise ValueError("evaluation question IDs must be unique")
    return questions


def passage_id(result: RetrievalResult) -> str:
    """Return the canonical passage ID required for verse-level evaluation."""
    value = result.document.metadata.get("passage_id")
    if not isinstance(value, str) or not value:
        raise ValueError("retrieval result is missing metadata.passage_id")
    return value


def evaluate_results(question: EvaluationQuestion, results: Sequence[RetrievalResult], k: int) -> RetrievalEvaluation:
    """Calculate verse-level metrics so multiple layers cannot inflate relevance."""
    if k < 1:
        raise ValueError("k must be positive")
    top_results = list(results[:k])
    retrieved_ids = [passage_id(result) for result in top_results]
    unique_ids = set(retrieved_ids)
    expected_found = question.expected_passage_ids & unique_ids
    first_relevant_rank = next(
        (rank for rank, result_id in enumerate(retrieved_ids, start=1) if result_id in question.expected_passage_ids),
        None,
    )
    chapters = {
        result.document.metadata.get("chapter")
        for result in top_results
        if isinstance(result.document.metadata.get("chapter"), int)
    }
    return RetrievalEvaluation(
        recall_at_k=len(expected_found) / len(question.expected_passage_ids),
        precision_at_k=len(expected_found) / k,
        reciprocal_rank=0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank,
        duplicate_result_rate=0.0 if not top_results else 1.0 - (len(unique_ids) / len(top_results)),
        unique_passage_count=len(unique_ids),
        repeated_passage_count=len(top_results) - len(unique_ids),
        chapter_count=len(chapters),
        expected_found=frozenset(expected_found),
    )
