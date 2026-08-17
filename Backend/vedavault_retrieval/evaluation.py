"""Deterministic, model-independent retrieval benchmark loading and scoring."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re

from .language import SupportedLanguage, WritingScript
from .retrieval import RetrievalResult


BENCHMARK_VERSION = 1
CANONICAL_PASSAGE_ID_PATTERN = re.compile(r"^BG_\d{2}_\d{2}$")
BENCHMARK_ID_PATTERN = re.compile(r"^RET_[A-Z]{2}_[A-Z0-9_]+$")

CATEGORIES = frozenset(
    {
        "action-and-results",
        "desire-and-anger",
        "devotion",
        "duty-and-responsibility",
        "knowledge-and-wisdom",
        "liberation",
        "mind-and-meditation",
        "relationships-and-ethics",
        "self-and-mortality",
        "success-and-failure",
        "suffering-grief-and-fear",
    }
)
DIFFICULTY_TAGS = frozenset(
    {
        "code-switched",
        "cross-lingual",
        "direct",
        "imperfect-input",
        "multi-verse",
        "philosophical",
        "semantic",
        "transliterated",
    }
)
BREAKDOWN_DIMENSIONS = frozenset(
    {
        "category",
        "code_switched",
        "cross_language",
        "difficulty",
        "imperfect_input",
        "input_language",
        "transliterated",
    }
)


@dataclass(frozen=True)
class EvaluationQuestion:
    """A question with canonical primary and acceptable verse judgments.

    The original ``expected_passage_ids`` and summary metric aliases remain for
    compatibility with the earlier small evaluation set. New benchmark items
    use ``acceptable_passage_ids`` as the complete relevance set, including
    all primary verses.
    """

    question_id: str
    question: str
    expected_passage_ids: frozenset[str]
    primary_passage_ids: frozenset[str] = frozenset()
    acceptable_passage_ids: frozenset[str] = frozenset()
    rationale: str = ""
    category: str | None = None
    input_language: SupportedLanguage | None = None
    writing_script: WritingScript | None = None
    transliterated: bool = False
    code_switched: bool = False
    imperfect_input: bool = False
    cross_language: bool = False
    difficulty_tags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.question_id, str) or not self.question_id:
            raise ValueError("evaluation questions require a non-empty id")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("evaluation questions require a non-empty question")
        expected = frozenset(self.expected_passage_ids)
        if not expected:
            raise ValueError("evaluation questions require expected passage IDs")
        if not all(isinstance(passage_id, str) and passage_id for passage_id in expected):
            raise ValueError("expected passage IDs must be non-empty strings")
        primary = frozenset(self.primary_passage_ids) or expected
        acceptable = frozenset(self.acceptable_passage_ids) or expected
        # Legacy callers supplied acceptable alternatives without repeating the
        # primary set. Normalize that old representation at this boundary.
        acceptable |= primary
        if not primary <= acceptable:
            raise ValueError("primary passage IDs must be included in acceptable passage IDs")
        for field_name in ("transliterated", "code_switched", "imperfect_input", "cross_language"):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be boolean")
        if self.category is not None and self.category not in CATEGORIES:
            raise ValueError("category must be a controlled benchmark category")
        if self.input_language is not None and not isinstance(self.input_language, SupportedLanguage):
            raise ValueError("input_language must be a SupportedLanguage or None")
        if self.writing_script is not None and not isinstance(self.writing_script, WritingScript):
            raise ValueError("writing_script must be a WritingScript or None")
        difficulty_tags = frozenset(self.difficulty_tags)
        if not difficulty_tags <= DIFFICULTY_TAGS:
            raise ValueError("difficulty_tags must be controlled benchmark difficulty tags")
        object.__setattr__(self, "expected_passage_ids", acceptable)
        object.__setattr__(self, "primary_passage_ids", primary)
        object.__setattr__(self, "acceptable_passage_ids", acceptable)
        object.__setattr__(self, "difficulty_tags", difficulty_tags)

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic representation of a fully specified benchmark item."""
        if self.category is None or self.input_language is None or self.writing_script is None:
            raise ValueError("legacy evaluation questions do not have complete benchmark metadata")
        return {
            "id": self.question_id,
            "category": self.category,
            "query": self.question,
            "input_language": self.input_language.value,
            "writing_script": self.writing_script.value,
            "transliterated": self.transliterated,
            "code_switched": self.code_switched,
            "imperfect_input": self.imperfect_input,
            "cross_language": self.cross_language,
            "difficulty": sorted(self.difficulty_tags),
            "primary_verses": sorted(self.primary_passage_ids),
            "acceptable_verses": sorted(self.acceptable_passage_ids),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class RetrievalBenchmark:
    """Versioned, immutable collection of corpus-verified evaluation items."""

    benchmark_version: int
    work: str
    questions: tuple[EvaluationQuestion, ...]

    def __post_init__(self) -> None:
        if self.benchmark_version != BENCHMARK_VERSION:
            raise ValueError(f"unsupported benchmark_version: {self.benchmark_version}")
        if not isinstance(self.work, str) or not self.work.strip():
            raise ValueError("benchmark work must be a non-empty string")
        questions = tuple(self.questions)
        if not questions:
            raise ValueError("benchmark must contain at least one question")
        if len({question.question_id for question in questions}) != len(questions):
            raise ValueError("benchmark question IDs must be unique")
        object.__setattr__(self, "questions", questions)

    def to_dict(self) -> dict[str, object]:
        return {
            "benchmark_version": self.benchmark_version,
            "work": self.work,
            "items": [question.to_dict() for question in self.questions],
        }


@dataclass(frozen=True)
class RetrievalEvaluation:
    """Verse-level relevance metrics plus top-ten diversity diagnostics."""

    primary_recall_at_k: float
    acceptable_recall_at_k: float
    acceptable_precision_at_k: float
    primary_reciprocal_rank: float
    acceptable_reciprocal_rank: float
    duplicate_result_rate: float
    unique_passage_count: int
    repeated_passage_count: int
    chapter_count: int
    primary_found: frozenset[str]
    acceptable_found: frozenset[str]
    acceptable_only_found: frozenset[str]
    metric_cutoff: int
    diagnostic_cutoff: int
    diagnostic_result_count: int

    @property
    def recall_at_k(self) -> float:
        """Legacy alias for acceptable-set recall at K."""
        return self.acceptable_recall_at_k

    @property
    def precision_at_k(self) -> float:
        """Legacy alias for acceptable precision at K."""
        return self.acceptable_precision_at_k

    @property
    def reciprocal_rank(self) -> float:
        """Legacy alias for acceptable MRR."""
        return self.acceptable_reciprocal_rank

    @property
    def expected_found(self) -> frozenset[str]:
        """Legacy alias for acceptable hits within the metric cutoff."""
        return self.acceptable_found


@dataclass(frozen=True)
class AggregateRetrievalEvaluation:
    """Deterministic mean metrics with an explicit subgroup sample count."""

    sample_count: int
    mean_primary_recall_at_k: float
    mean_acceptable_recall_at_k: float
    mean_acceptable_precision_at_k: float
    mean_primary_reciprocal_rank: float
    mean_acceptable_reciprocal_rank: float
    mean_duplicate_result_rate: float
    mean_unique_passage_count: float
    mean_chapter_count: float


def load_retrieval_benchmark(path: Path) -> RetrievalBenchmark:
    """Load and validate a versioned benchmark without loading retrieval models."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or set(value) != {"benchmark_version", "work", "items"}:
        raise ValueError("benchmark must contain benchmark_version, work, and items")
    if not isinstance(value["benchmark_version"], int):
        raise ValueError("benchmark_version must be an integer")
    if not isinstance(value["work"], str) or not value["work"].strip():
        raise ValueError("work must be a non-empty string")
    if not isinstance(value["items"], list):
        raise ValueError("benchmark items must be a JSON list")
    return RetrievalBenchmark(value["benchmark_version"], value["work"], tuple(_question_from_dict(item) for item in value["items"]))


def load_evaluation_questions(path: Path) -> list[EvaluationQuestion]:
    """Compatibility loader returning the ordered questions from a benchmark file."""
    return list(load_retrieval_benchmark(path).questions)


def validate_benchmark_corpus_references(benchmark: RetrievalBenchmark, corpus_path: Path) -> None:
    """Verify every curated canonical verse ID exists in the local canonical corpus."""
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    if not isinstance(corpus, Mapping) or not isinstance(corpus.get("passages"), list):
        raise ValueError("canonical corpus must contain a passages list")
    canonical_ids = {
        passage.get("passage_id")
        for passage in corpus["passages"]
        if isinstance(passage, Mapping) and isinstance(passage.get("passage_id"), str)
    }
    missing = sorted(
        {
            passage_id
            for question in benchmark.questions
            for passage_id in question.acceptable_passage_ids
            if passage_id not in canonical_ids
        }
    )
    if missing:
        raise ValueError(f"benchmark references verses absent from the canonical corpus: {', '.join(missing)}")


def passage_id(result: RetrievalResult) -> str:
    """Return the canonical passage ID required for verse-level evaluation."""
    value = result.document.metadata.get("passage_id")
    if not isinstance(value, str) or not value:
        raise ValueError("retrieval result is missing metadata.passage_id")
    return value


def evaluate_results(
    question: EvaluationQuestion,
    results: Sequence[RetrievalResult],
    k: int,
    *,
    diagnostic_k: int = 10,
) -> RetrievalEvaluation:
    """Score one ranked result list without changing its ranks or deduplicating it.

    Recall uses unique canonical verse IDs in the first ``k`` positions.
    Acceptable precision uses ``k`` as its fixed denominator even when fewer
    results are supplied. MRR uses the first primary or acceptable hit in the
    supplied ranking. Duplicate, unique-verse, and chapter diagnostics use the
    first ``diagnostic_k`` positions (ten by default).
    """
    if k < 1 or diagnostic_k < 1:
        raise ValueError("k and diagnostic_k must be positive")
    ranked_results = list(results)
    top_results = ranked_results[:k]
    retrieved_ids = [passage_id(result) for result in top_results]
    unique_ids = set(retrieved_ids)
    primary_found = question.primary_passage_ids & unique_ids
    acceptable_found = question.acceptable_passage_ids & unique_ids
    diagnostic_results = ranked_results[:diagnostic_k]
    diagnostic_ids = [passage_id(result) for result in diagnostic_results]
    unique_diagnostic_ids = set(diagnostic_ids)
    chapters = {
        result.document.metadata.get("chapter")
        for result in diagnostic_results
        if isinstance(result.document.metadata.get("chapter"), int)
    }
    return RetrievalEvaluation(
        primary_recall_at_k=len(primary_found) / len(question.primary_passage_ids),
        acceptable_recall_at_k=len(acceptable_found) / len(question.acceptable_passage_ids),
        acceptable_precision_at_k=len(acceptable_found) / k,
        primary_reciprocal_rank=_reciprocal_rank(ranked_results, question.primary_passage_ids),
        acceptable_reciprocal_rank=_reciprocal_rank(ranked_results, question.acceptable_passage_ids),
        duplicate_result_rate=0.0
        if not diagnostic_results
        else 1.0 - (len(unique_diagnostic_ids) / len(diagnostic_results)),
        unique_passage_count=len(unique_diagnostic_ids),
        repeated_passage_count=len(diagnostic_results) - len(unique_diagnostic_ids),
        chapter_count=len(chapters),
        primary_found=frozenset(primary_found),
        acceptable_found=frozenset(acceptable_found),
        acceptable_only_found=frozenset(acceptable_found - question.primary_passage_ids),
        metric_cutoff=k,
        diagnostic_cutoff=diagnostic_k,
        diagnostic_result_count=len(diagnostic_results),
    )


def aggregate_evaluations(
    scored_questions: Sequence[tuple[EvaluationQuestion, RetrievalEvaluation]],
    *,
    by: str,
) -> dict[str, AggregateRetrievalEvaluation]:
    """Aggregate already-scored benchmark questions by explicit metadata only."""
    if by not in BREAKDOWN_DIMENSIONS:
        raise ValueError(f"unsupported breakdown dimension: {by}")
    groups: dict[str, list[RetrievalEvaluation]] = defaultdict(list)
    for question, evaluation in scored_questions:
        for key in _breakdown_keys(question, by):
            groups[key].append(evaluation)
    return {key: _aggregate(groups[key]) for key in sorted(groups)}


def _question_from_dict(item: object) -> EvaluationQuestion:
    if not isinstance(item, Mapping):
        raise ValueError("each benchmark item must be a JSON object")
    required = {
        "id",
        "category",
        "query",
        "input_language",
        "writing_script",
        "transliterated",
        "code_switched",
        "imperfect_input",
        "cross_language",
        "difficulty",
        "primary_verses",
        "acceptable_verses",
        "rationale",
    }
    if set(item) != required:
        raise ValueError("benchmark items must contain the complete version-1 schema")
    question_id = item["id"]
    if not isinstance(question_id, str) or not BENCHMARK_ID_PATTERN.fullmatch(question_id):
        raise ValueError("benchmark item id must use the stable RET_<LANG>_<NAME> convention")
    category = item["category"]
    if not isinstance(category, str) or category not in CATEGORIES:
        raise ValueError("benchmark item category is not controlled")
    query = item["query"]
    if not isinstance(query, str) or not query.strip():
        raise ValueError("benchmark item query must be a non-empty string")
    input_language = _enum_value(SupportedLanguage, item["input_language"], "input_language")
    writing_script = _enum_value(WritingScript, item["writing_script"], "writing_script")
    booleans = {}
    for field_name in ("transliterated", "code_switched", "imperfect_input", "cross_language"):
        value = item[field_name]
        if not isinstance(value, bool):
            raise ValueError(f"benchmark item {field_name} must be boolean")
        booleans[field_name] = value
    difficulty = item["difficulty"]
    if not isinstance(difficulty, list) or not difficulty or not all(isinstance(tag, str) for tag in difficulty):
        raise ValueError("benchmark item difficulty must be a non-empty list of strings")
    if len(set(difficulty)) != len(difficulty) or not set(difficulty) <= DIFFICULTY_TAGS:
        raise ValueError("benchmark item difficulty must use unique controlled tags")
    required_tags = {
        "transliterated": "transliterated",
        "code_switched": "code-switched",
        "imperfect_input": "imperfect-input",
        "cross_language": "cross-lingual",
    }
    if any(booleans[field_name] and tag not in difficulty for field_name, tag in required_tags.items()):
        raise ValueError("benchmark difficulty must declare each enabled metadata characteristic")
    primary = _verse_ids(item["primary_verses"], "primary_verses")
    acceptable = _verse_ids(item["acceptable_verses"], "acceptable_verses")
    if not primary <= acceptable:
        raise ValueError("benchmark primary_verses must be included in acceptable_verses")
    rationale = item["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("benchmark item rationale must be a non-empty string")
    return EvaluationQuestion(
        question_id,
        query,
        acceptable,
        primary,
        acceptable,
        rationale,
        category,
        input_language,
        writing_script,
        booleans["transliterated"],
        booleans["code_switched"],
        booleans["imperfect_input"],
        booleans["cross_language"],
        frozenset(difficulty),
    )


def _enum_value(enum_type: type[SupportedLanguage] | type[WritingScript], value: object, field_name: str):
    if not isinstance(value, str):
        raise ValueError(f"benchmark item {field_name} must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ValueError(f"benchmark item {field_name} is unsupported") from error


def _verse_ids(value: object, field_name: str) -> frozenset[str]:
    if not isinstance(value, list) or not value or not all(isinstance(passage_id, str) for passage_id in value):
        raise ValueError(f"benchmark item {field_name} must be a non-empty list of verse IDs")
    if len(set(value)) != len(value):
        raise ValueError(f"benchmark item {field_name} must not contain duplicate verse IDs")
    if not all(CANONICAL_PASSAGE_ID_PATTERN.fullmatch(passage_id) for passage_id in value):
        raise ValueError(f"benchmark item {field_name} contains an invalid canonical verse ID")
    return frozenset(value)


def _reciprocal_rank(results: Sequence[RetrievalResult], targets: frozenset[str]) -> float:
    for rank, result in enumerate(results, start=1):
        if passage_id(result) in targets:
            return 1.0 / rank
    return 0.0


def _breakdown_keys(question: EvaluationQuestion, by: str) -> tuple[str, ...]:
    if by == "difficulty":
        return tuple(sorted(question.difficulty_tags))
    value = getattr(question, by)
    if isinstance(value, SupportedLanguage):
        return (value.value,)
    if isinstance(value, bool):
        return (str(value).lower(),)
    if isinstance(value, str):
        return (value,)
    raise ValueError(f"question {question.question_id} has no {by} metadata for aggregation")


def _aggregate(evaluations: Sequence[RetrievalEvaluation]) -> AggregateRetrievalEvaluation:
    if not evaluations:
        raise ValueError("cannot aggregate an empty evaluation group")
    count = len(evaluations)
    return AggregateRetrievalEvaluation(
        sample_count=count,
        mean_primary_recall_at_k=sum(item.primary_recall_at_k for item in evaluations) / count,
        mean_acceptable_recall_at_k=sum(item.acceptable_recall_at_k for item in evaluations) / count,
        mean_acceptable_precision_at_k=sum(item.acceptable_precision_at_k for item in evaluations) / count,
        mean_primary_reciprocal_rank=sum(item.primary_reciprocal_rank for item in evaluations) / count,
        mean_acceptable_reciprocal_rank=sum(item.acceptable_reciprocal_rank for item in evaluations) / count,
        mean_duplicate_result_rate=sum(item.duplicate_result_rate for item in evaluations) / count,
        mean_unique_passage_count=sum(item.unique_passage_count for item in evaluations) / count,
        mean_chapter_count=sum(item.chapter_count for item in evaluations) / count,
    )
