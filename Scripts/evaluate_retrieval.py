"""Evaluate a local VedaVault index against version-controlled retrieval questions."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Backend"))

from vedavault_retrieval import (  # noqa: E402
    LocalVectorStore,
    Retriever,
    SentenceTransformerEmbeddingProvider,
    deduplicate_by_passage,
    evaluate_results,
    load_evaluation_questions,
)


INDEX_DIRECTORY = ROOT / "Data" / "Processed" / "Bhagavad_Gita" / "retrieval_index"
EVALUATION_DATASET = ROOT / "Evaluation" / "bhagavad_gita_retrieval.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5, help="results evaluated per question (default: 5)")
    parser.add_argument(
        "--deduplicate-by-verse",
        action="store_true",
        help="select one highest-ranked result per verse before scoring final top-K context",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=25,
        help="raw candidates considered when verse diversification is enabled (default: 25)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.k < 1 or args.candidate_limit < args.k:
        raise SystemExit("--k must be positive and --candidate-limit must be at least --k")
    if not INDEX_DIRECTORY.is_dir():
        raise SystemExit("Index not found. Build it before running retrieval evaluation.")

    # Evaluation must not initiate a model download; it only uses already-cached local weights.
    provider = SentenceTransformerEmbeddingProvider(local_files_only=True)
    store = LocalVectorStore.load(INDEX_DIRECTORY, embedding_provider=provider)
    retriever = Retriever(provider, store)
    questions = load_evaluation_questions(EVALUATION_DATASET)
    all_metrics = []
    raw_metrics_all = []

    for question in questions:
        raw_results = retriever.retrieve(question.question, limit=args.candidate_limit)
        results = (
            deduplicate_by_passage(raw_results)[: args.k]
            if args.deduplicate_by_verse
            else raw_results[: args.k]
        )
        metrics = evaluate_results(question, results, args.k)
        raw_metrics = evaluate_results(question, raw_results, args.k)
        all_metrics.append(metrics)
        raw_metrics_all.append(raw_metrics)
        repeated_layers: dict[str, list[str]] = defaultdict(list)
        for result in raw_results[: args.k]:
            repeated_layers[result.document.metadata["passage_id"]].append(result.document.metadata["text_layer"])
        repeated_summary = "; ".join(
            f"{passage_id} ({', '.join(layers)})"
            for passage_id, layers in repeated_layers.items()
            if len(layers) > 1
        )

        print(f"\nQuestion: {question.question}")
        print(f"Primary Recall@{args.k}: {metrics.primary_recall_at_k:.3f}")
        print(f"Acceptable Recall@{args.k}: {metrics.acceptable_recall_at_k:.3f}")
        print(f"Acceptable Precision@{args.k}: {metrics.acceptable_precision_at_k:.3f}")
        print(f"Primary MRR: {metrics.primary_reciprocal_rank:.3f}")
        print(f"Acceptable MRR: {metrics.acceptable_reciprocal_rank:.3f}")
        print(
            "Duplicates: "
            f"{metrics.repeated_passage_count}/{len(results)} final "
            f"({metrics.duplicate_result_rate:.1%}); raw top-{args.k}: {raw_metrics.repeated_passage_count}"
        )
        print(f"Diversity: {metrics.unique_passage_count} unique verses across {metrics.chapter_count} chapters")
        print(f"Raw repeated verse/layers: {repeated_summary or 'none'}")
        print(f"Primary present: {', '.join(sorted(metrics.primary_found)) or 'none'}")
        print(f"Acceptable-only present: {', '.join(sorted(metrics.acceptable_only_found)) or 'none'}")
        print("Retrieved:")
        for result in results:
            metadata = result.document.metadata
            print(
                f"  {metadata['passage_id']} ({metadata['chapter']}:{metadata['verse']}) "
                f"layer={metadata['text_layer']} score={result.score:.4f}"
            )
        print(f"Primary: {', '.join(sorted(question.primary_passage_ids))}")
        print(f"Acceptable: {', '.join(sorted(question.acceptable_passage_ids))}")

    count = len(all_metrics)
    print("\n=== Aggregate ===")
    print(f"Questions: {count}; mode: {'deduplicated by verse' if args.deduplicate_by_verse else 'raw ranked layers'}")
    print(f"Mean Primary Recall@{args.k}: {sum(metric.primary_recall_at_k for metric in all_metrics) / count:.3f}")
    print(f"Mean Acceptable Recall@{args.k}: {sum(metric.acceptable_recall_at_k for metric in all_metrics) / count:.3f}")
    print(f"Mean Acceptable Precision@{args.k}: {sum(metric.acceptable_precision_at_k for metric in all_metrics) / count:.3f}")
    print(f"Mean Primary MRR: {sum(metric.primary_reciprocal_rank for metric in all_metrics) / count:.3f}")
    print(f"Mean Acceptable MRR: {sum(metric.acceptable_reciprocal_rank for metric in all_metrics) / count:.3f}")
    print(f"Mean duplicate-result rate: {sum(metric.duplicate_result_rate for metric in all_metrics) / count:.1%}")
    print(f"Mean raw top-{args.k} duplicate-result rate: {sum(metric.duplicate_result_rate for metric in raw_metrics_all) / count:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
