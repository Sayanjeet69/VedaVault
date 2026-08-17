"""Evaluate a local VedaVault index without changing raw retrieval rank semantics."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Backend"))

from vedavault_retrieval import (  # noqa: E402
    EvaluationQuestion,
    LocalVectorStore,
    RetrievalEvaluation,
    RetrievalResult,
    Retriever,
    SentenceTransformerEmbeddingProvider,
    deduplicate_by_passage,
    evaluate_results,
    load_evaluation_questions,
)


INDEX_DIRECTORY = ROOT / "Data" / "Processed" / "Bhagavad_Gita" / "retrieval_index"
EVALUATION_DATASET = ROOT / "Evaluation" / "bhagavad_gita_retrieval.json"
RAW_RECALL_CUTOFFS = (5, 10)


@dataclass(frozen=True)
class DiversifiedContextDiagnostics:
    """Non-ranking diagnostics for an optional deduplicated downstream context."""

    context_size: int
    duplicate_result_rate: float
    unique_passage_count: int
    repeated_passage_count: int
    chapter_count: int


@dataclass(frozen=True)
class QuestionEvaluation:
    """Raw retrieval metrics plus optional, explicitly separate context diagnostics."""

    raw_at_5: RetrievalEvaluation
    raw_at_10: RetrievalEvaluation
    context_results: tuple[RetrievalResult, ...]
    diversified_context: DiversifiedContextDiagnostics | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context-k",
        "--k",
        dest="context_k",
        type=int,
        default=5,
        help="results selected for a future downstream context (default: 5)",
    )
    parser.add_argument(
        "--diagnostic-k",
        type=int,
        default=10,
        help="raw ranked positions inspected for duplicate/diversity diagnostics (default: 10)",
    )
    parser.add_argument(
        "--deduplicate-by-verse",
        action="store_true",
        help="also report diagnostics for a highest-ranked-per-verse final context",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=25,
        help="raw candidates retrieved; must cover Recall@10 and diagnostic positions (default: 25)",
    )
    return parser.parse_args()


def validate_limits(context_k: int, diagnostic_k: int, candidate_limit: int) -> None:
    """Ensure raw ranking contains every position needed for reported metrics."""
    if context_k < 1:
        raise ValueError("--context-k must be positive")
    if diagnostic_k < 10:
        raise ValueError("--diagnostic-k must be at least 10 because raw Recall@10 is reported")
    if candidate_limit < max(context_k, diagnostic_k, max(RAW_RECALL_CUTOFFS)):
        raise ValueError("--candidate-limit must cover context-k, diagnostic-k, and raw Recall@10")


def evaluate_question(
    question: EvaluationQuestion,
    raw_results: Sequence[RetrievalResult],
    *,
    context_k: int,
    diagnostic_k: int,
    deduplicate_by_verse: bool,
) -> QuestionEvaluation:
    """Score raw results before optional context diversification.

    Raw recall, precision, MRR, duplicate rate, unique-verse count, and
    chapter diversity always use the original supplied ranking. Diversification
    is only a downstream-context selection and never supplies MRR.
    """
    raw_results = tuple(raw_results)
    raw_at_5 = evaluate_results(question, raw_results, 5, diagnostic_k=diagnostic_k)
    raw_at_10 = evaluate_results(question, raw_results, 10, diagnostic_k=diagnostic_k)
    if not deduplicate_by_verse:
        return QuestionEvaluation(raw_at_5, raw_at_10, raw_results[:context_k], None)

    context_results = tuple(deduplicate_by_passage(list(raw_results))[:context_k])
    return QuestionEvaluation(
        raw_at_5,
        raw_at_10,
        context_results,
        _context_diagnostics(context_results),
    )


def _context_diagnostics(results: Sequence[RetrievalResult]) -> DiversifiedContextDiagnostics:
    """Describe a final context without treating it as an original ranking."""
    result_list = tuple(results)
    passage_ids = [result.document.metadata.get("passage_id") for result in result_list]
    if not all(isinstance(passage_id, str) and passage_id for passage_id in passage_ids):
        raise ValueError("context result is missing metadata.passage_id")
    unique_passage_ids = set(passage_ids)
    chapters = {
        result.document.metadata.get("chapter")
        for result in result_list
        if isinstance(result.document.metadata.get("chapter"), int)
    }
    return DiversifiedContextDiagnostics(
        context_size=len(result_list),
        duplicate_result_rate=0.0 if not result_list else 1.0 - (len(unique_passage_ids) / len(result_list)),
        unique_passage_count=len(unique_passage_ids),
        repeated_passage_count=len(result_list) - len(unique_passage_ids),
        chapter_count=len(chapters),
    )


def main() -> int:
    args = parse_args()
    try:
        validate_limits(args.context_k, args.diagnostic_k, args.candidate_limit)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not INDEX_DIRECTORY.is_dir():
        raise SystemExit("Index not found. Build it before running retrieval evaluation.")

    # Evaluation must not initiate a model download; it only uses already-cached local weights.
    provider = SentenceTransformerEmbeddingProvider(local_files_only=True)
    store = LocalVectorStore.load(INDEX_DIRECTORY, embedding_provider=provider)
    retriever = Retriever(provider, store)
    questions = load_evaluation_questions(EVALUATION_DATASET)
    evaluations: list[QuestionEvaluation] = []

    for question in questions:
        raw_results = retriever.retrieve(question.question, limit=args.candidate_limit)
        evaluation = evaluate_question(
            question,
            raw_results,
            context_k=args.context_k,
            diagnostic_k=args.diagnostic_k,
            deduplicate_by_verse=args.deduplicate_by_verse,
        )
        evaluations.append(evaluation)
        _print_question_report(question, raw_results, evaluation, args)

    _print_aggregate_report(evaluations, args)
    return 0


def _print_question_report(
    question: EvaluationQuestion,
    raw_results: Sequence[RetrievalResult],
    evaluation: QuestionEvaluation,
    args: argparse.Namespace,
) -> None:
    raw_at_5, raw_at_10 = evaluation.raw_at_5, evaluation.raw_at_10
    repeated_layers: dict[str, list[str]] = defaultdict(list)
    for result in raw_results[: args.diagnostic_k]:
        repeated_layers[result.document.metadata["passage_id"]].append(result.document.metadata["text_layer"])
    repeated_summary = "; ".join(
        f"{passage_id} ({', '.join(layers)})"
        for passage_id, layers in repeated_layers.items()
        if len(layers) > 1
    )

    print(f"\nQuestion: {question.question}")
    print("Raw retrieval metrics (original ranking):")
    print(f"  Primary Recall@5: {raw_at_5.primary_recall_at_k:.3f}")
    print(f"  Primary Recall@10: {raw_at_10.primary_recall_at_k:.3f}")
    print(f"  Acceptable Recall@5: {raw_at_5.acceptable_recall_at_k:.3f}")
    print(f"  Acceptable Recall@10: {raw_at_10.acceptable_recall_at_k:.3f}")
    print(f"  Acceptable Precision@5: {raw_at_5.acceptable_precision_at_k:.3f}")
    print(f"  Primary MRR: {raw_at_5.primary_reciprocal_rank:.3f}")
    print(f"  Acceptable MRR: {raw_at_5.acceptable_reciprocal_rank:.3f}")
    print(
        f"  Raw diagnostics@{args.diagnostic_k}: {raw_at_5.repeated_passage_count}/"
        f"{raw_at_5.diagnostic_result_count} duplicates ({raw_at_5.duplicate_result_rate:.1%}); "
        f"{raw_at_5.unique_passage_count} unique verses across {raw_at_5.chapter_count} chapters"
    )
    print(f"  Raw repeated verse/layers: {repeated_summary or 'none'}")
    print(f"  Primary present@5: {', '.join(sorted(raw_at_5.primary_found)) or 'none'}")
    print(f"  Primary present@10: {', '.join(sorted(raw_at_10.primary_found)) or 'none'}")
    print(f"  Acceptable-only present@10: {', '.join(sorted(raw_at_10.acceptable_only_found)) or 'none'}")
    if evaluation.diversified_context is not None:
        context = evaluation.diversified_context
        print(
            "Diversified context diagnostics (not retrieval metrics): "
            f"{context.context_size} results; {context.repeated_passage_count}/{context.context_size} duplicates "
            f"({context.duplicate_result_rate:.1%}); {context.unique_passage_count} unique verses across "
            f"{context.chapter_count} chapters"
        )
    print("Final context:")
    for result in evaluation.context_results:
        metadata = result.document.metadata
        print(
            f"  {metadata['passage_id']} ({metadata['chapter']}:{metadata['verse']}) "
            f"layer={metadata['text_layer']} score={result.score:.4f}"
        )
    print(f"Primary: {', '.join(sorted(question.primary_passage_ids))}")
    print(f"Acceptable: {', '.join(sorted(question.acceptable_passage_ids))}")


def _print_aggregate_report(evaluations: Sequence[QuestionEvaluation], args: argparse.Namespace) -> None:
    count = len(evaluations)
    raw_at_5 = [evaluation.raw_at_5 for evaluation in evaluations]
    raw_at_10 = [evaluation.raw_at_10 for evaluation in evaluations]
    print("\n=== Aggregate raw retrieval metrics (original ranking) ===")
    print(f"Questions: {count}; context mode: {'diversified by verse' if args.deduplicate_by_verse else 'raw ranked layers'}")
    print(f"Mean Primary Recall@5: {_mean(metric.primary_recall_at_k for metric in raw_at_5):.3f}")
    print(f"Mean Primary Recall@10: {_mean(metric.primary_recall_at_k for metric in raw_at_10):.3f}")
    print(f"Mean Acceptable Recall@5: {_mean(metric.acceptable_recall_at_k for metric in raw_at_5):.3f}")
    print(f"Mean Acceptable Recall@10: {_mean(metric.acceptable_recall_at_k for metric in raw_at_10):.3f}")
    print(f"Mean Acceptable Precision@5: {_mean(metric.acceptable_precision_at_k for metric in raw_at_5):.3f}")
    print(f"Mean Primary MRR: {_mean(metric.primary_reciprocal_rank for metric in raw_at_5):.3f}")
    print(f"Mean Acceptable MRR: {_mean(metric.acceptable_reciprocal_rank for metric in raw_at_5):.3f}")
    print(f"Mean raw duplicate-result rate@{args.diagnostic_k}: {_mean(metric.duplicate_result_rate for metric in raw_at_5):.1%}")
    print(f"Mean raw unique verses@{args.diagnostic_k}: {_mean(metric.unique_passage_count for metric in raw_at_5):.2f}")
    print(f"Mean raw chapter diversity@{args.diagnostic_k}: {_mean(metric.chapter_count for metric in raw_at_5):.2f}")
    diversified = [evaluation.diversified_context for evaluation in evaluations if evaluation.diversified_context is not None]
    if diversified:
        print("\n=== Aggregate diversified context diagnostics (not retrieval metrics) ===")
        print(f"Mean context size: {_mean(item.context_size for item in diversified):.2f}")
        print(f"Mean context duplicate-result rate: {_mean(item.duplicate_result_rate for item in diversified):.1%}")
        print(f"Mean context unique verses: {_mean(item.unique_passage_count for item in diversified):.2f}")
        print(f"Mean context chapter diversity: {_mean(item.chapter_count for item in diversified):.2f}")


def _mean(values: Iterable[float]) -> float:
    values = tuple(values)
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
