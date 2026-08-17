"""Diagnose local Bhagavad Gita retrieval quality without rebuilding its index."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Backend"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vedavault_retrieval import (  # noqa: E402
    LocalVectorStore,
    Retriever,
    SentenceTransformerEmbeddingProvider,
    evaluate_results,
    load_evaluation_questions,
)


INDEX_DIRECTORY = ROOT / "Data" / "Processed" / "Bhagavad_Gita" / "retrieval_index"
EVALUATION_DATASET = ROOT / "Evaluation" / "bhagavad_gita_retrieval.json"
TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def _document_profiles(index_directory: Path) -> tuple[Counter[str], dict[str, list[dict[str, object]]]]:
    """Read persisted metadata only; no corpus or index files are changed."""
    documents = json.loads((index_directory / "documents.json").read_text(encoding="utf-8"))
    layers: Counter[str] = Counter()
    by_passage: dict[str, list[dict[str, object]]] = defaultdict(list)
    for document in documents:
        metadata = document["metadata"]
        layers[metadata["text_layer"]] += 1
        by_passage[metadata["passage_id"]].append(document)
    return layers, by_passage


def _normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _is_substantially_different(left: str, right: str) -> bool:
    left_terms, right_terms = set(TOKEN_PATTERN.findall(left.casefold())), set(TOKEN_PATTERN.findall(right.casefold()))
    if not left_terms or not right_terms:
        return left != right
    return len(left_terms & right_terms) / len(left_terms | right_terms) < 0.25


def _print_representation_summary(by_passage: dict[str, list[dict[str, object]]], layers: Counter[str]) -> None:
    layer_combinations: Counter[tuple[str, ...]] = Counter()
    exact_duplicate_verses = 0
    substantially_different_verses = 0
    for documents in by_passage.values():
        layer_combinations[tuple(sorted({document["metadata"]["text_layer"] for document in documents}))] += 1
        texts = list({_normalized_text(document["text"]) for document in documents})
        if len(texts) < len(documents):
            exact_duplicate_verses += 1
        if any(_is_substantially_different(left, right) for index, left in enumerate(texts) for right in texts[index + 1 :]):
            substantially_different_verses += 1

    print("=== Index layer and cross-layer representation ===")
    print(f"Indexed documents: {sum(layers.values())}; canonical verses represented: {len(by_passage)}")
    print("Documents by layer: " + ", ".join(f"{layer}={count}" for layer, count in sorted(layers.items())))
    print(f"Verses with an exact duplicate stored text: {exact_duplicate_verses}")
    print(f"Verses with at least two substantially different stored texts: {substantially_different_verses}")
    print("Layer combinations by verse:")
    for combination, count in layer_combinations.most_common():
        print(f"  {', '.join(combination)}: {count}")


def _score_range(results: list, count: int) -> str:
    selected = results[:count]
    if not selected:
        return "no results"
    return f"{selected[0].score:.4f} to {selected[-1].score:.4f}"


def main() -> int:
    if not INDEX_DIRECTORY.is_dir():
        raise SystemExit("Index not found. Build it before running retrieval diagnostics.")
    layers, by_passage = _document_profiles(INDEX_DIRECTORY)
    _print_representation_summary(by_passage, layers)

    # Strictly local cache use prevents diagnostic runs from downloading model weights.
    provider = SentenceTransformerEmbeddingProvider(local_files_only=True)
    store = LocalVectorStore.load(INDEX_DIRECTORY, embedding_provider=provider)
    retriever = Retriever(provider, store)
    questions = load_evaluation_questions(EVALUATION_DATASET)
    top_ten_layers: Counter[str] = Counter()
    layer_metrics: dict[str, list] = defaultdict(list)

    print("\n=== Per-question semantic retrieval diagnostic ===")
    for question in questions:
        ranked = retriever.retrieve(question.question, limit=store.size)
        top_ten = ranked[:10]
        top_ten_layers.update(result.document.metadata["text_layer"] for result in top_ten)
        for layer in layers:
            layer_results = [result for result in ranked if result.document.metadata["text_layer"] == layer]
            layer_metrics[layer].append(evaluate_results(question, layer_results, 5))
        expected_ranked = [
            (rank, result)
            for rank, result in enumerate(ranked, start=1)
            if result.document.metadata["passage_id"] in question.expected_passage_ids
        ]

        print(f"\nQuestion: {question.question}")
        print("Expected: " + ", ".join(sorted(question.expected_passage_ids)))
        print(
            f"Scores: top-1={_score_range(ranked, 1)}; top-5={_score_range(ranked, 5)}; "
            f"top-10={_score_range(ranked, 10)}"
        )
        if expected_ranked:
            first_rank, first_expected = expected_ranked[0]
            gap = ranked[0].score - first_expected.score
            print(
                f"Best expected: {first_expected.document.metadata['passage_id']} at rank {first_rank} "
                f"score={first_expected.score:.4f}; top-1 gap={gap:.4f}"
            )
        else:
            print("Best expected: no indexed result found")
        print("Top 10:")
        for rank, result in enumerate(top_ten, start=1):
            metadata = result.document.metadata
            expected_marker = " expected" if metadata["passage_id"] in question.expected_passage_ids else ""
            snippet = " ".join(result.document.text.split())[:180]
            print(
                f"  {rank:2}. {metadata['passage_id']} layer={metadata['text_layer']} "
                f"score={result.score:.4f}{expected_marker}\n      {snippet}"
            )

    print("\n=== Top-10 layer dominance across evaluation questions ===")
    total = sum(top_ten_layers.values())
    for layer, count in top_ten_layers.most_common():
        print(f"{layer}: {count}/{total} ({count / total:.1%})")
    print("\n=== Layer-isolated evaluation at K=5 ===")
    for layer in sorted(layers):
        metrics = layer_metrics[layer]
        print(
            f"{layer}: Recall@5={sum(metric.recall_at_k for metric in metrics) / len(metrics):.3f}; "
            f"Precision@5={sum(metric.precision_at_k for metric in metrics) / len(metrics):.3f}; "
            f"MRR={sum(metric.reciprocal_rank for metric in metrics) / len(metrics):.3f}; "
            f"duplicate rate={sum(metric.duplicate_result_rate for metric in metrics) / len(metrics):.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
