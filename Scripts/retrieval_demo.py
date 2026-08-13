"""Print representative Bhagavad Gita retrieval results from the local index."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Backend"))

from vedavault_retrieval import LocalVectorStore, Retriever, SentenceTransformerEmbeddingProvider  # noqa: E402


INDEX_DIRECTORY = ROOT / "Data" / "Processed" / "Bhagavad_Gita" / "retrieval_index"
QUERIES = (
    "What does the Bhagavad Gita say about detachment from the fruits of action?",
    "What is karma yoga?",
    "What does Krishna say about the nature of the Self?",
)


def main() -> int:
    if not INDEX_DIRECTORY.is_dir():
        raise SystemExit("Index not found. Run: python Scripts/build_bhagavad_gita_index.py")
    retriever = Retriever(SentenceTransformerEmbeddingProvider(), LocalVectorStore.load(INDEX_DIRECTORY))
    for query in QUERIES:
        print(f"\nQuery: {query}")
        for result in retriever.retrieve(query, limit=3):
            metadata = result.document.metadata
            print(
                f"- {metadata['passage_id']} ({metadata['chapter']}:{metadata['verse']}) "
                f"score={result.score:.4f} layer={metadata['text_layer']}\n  {result.document.text[:260]}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
