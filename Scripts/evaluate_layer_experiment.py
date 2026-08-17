"""Compare all-layer and translations-only retrieval on the local evaluation set."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Backend"))

from vedavault_retrieval import (  # noqa: E402
    LocalVectorStore,
    Retriever,
    SentenceTransformerEmbeddingProvider,
    evaluate_results,
    load_evaluation_questions,
)


INDEX_DIRECTORY = ROOT / "Data" / "Processed" / "Bhagavad_Gita" / "retrieval_index"
EVALUATION_DATASET = ROOT / "Evaluation" / "bhagavad_gita_retrieval.json"
K = 5
CANDIDATE_LIMIT = 25


@dataclass(frozen=True)
class Configuration:
    name: str
    text_layers: tuple[str, ...] | None
    deduplicate_by_verse: bool


CONFIGURATIONS = (
    Configuration("A. all layers + normal", None, False),
    Configuration("B. all layers + verse diversification", None, True),
    Configuration("C. translations only + normal", ("translations",), False),
    Configuration("D. translations only + verse diversification", ("translations",), True),
)


def main() -> int:
    if not INDEX_DIRECTORY.is_dir():
        raise SystemExit("Index not found. Build it before running this experiment.")
    provider = SentenceTransformerEmbeddingProvider(local_files_only=True)
    store = LocalVectorStore.load(INDEX_DIRECTORY, embedding_provider=provider)
    retriever = Retriever(provider, store)
    questions = load_evaluation_questions(EVALUATION_DATASET)

    print(f"Layer-aware retrieval experiment: {len(questions)} questions, K={K}, candidate limit={CANDIDATE_LIMIT}")
    print("Relevance is primary verses plus documented acceptable alternatives.\n")
    print("Configuration | Recall@5 | Precision@5 | MRR | Duplicate verse rate | Unique verses | Chapter diversity")
    print("--- | ---: | ---: | ---: | ---: | ---: | ---:")
    for configuration in CONFIGURATIONS:
        metrics = []
        for question in questions:
            results = retriever.retrieve(
                question.question,
                limit=K,
                text_layers=configuration.text_layers,
                deduplicate_by_verse=configuration.deduplicate_by_verse,
                diversity_candidate_limit=CANDIDATE_LIMIT if configuration.deduplicate_by_verse else None,
            )
            metrics.append(evaluate_results(question, results, K))
        count = len(metrics)
        print(
            f"{configuration.name} | "
            f"{sum(metric.recall_at_k for metric in metrics) / count:.3f} | "
            f"{sum(metric.precision_at_k for metric in metrics) / count:.3f} | "
            f"{sum(metric.reciprocal_rank for metric in metrics) / count:.3f} | "
            f"{sum(metric.duplicate_result_rate for metric in metrics) / count:.1%} | "
            f"{sum(metric.unique_passage_count for metric in metrics) / count:.2f} | "
            f"{sum(metric.chapter_count for metric in metrics) / count:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
