"""Build the first local semantic index from the canonical Bhagavad Gita corpus."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Backend"))

from vedavault_retrieval import (  # noqa: E402
    LocalVectorStore,
    SentenceTransformerEmbeddingProvider,
    WordChunker,
    corpus_documents,
)


CORPUS_PATH = ROOT / "Data" / "Processed" / "Bhagavad_Gita" / "corpus.json"
INDEX_DIRECTORY = ROOT / "Data" / "Processed" / "Bhagavad_Gita" / "retrieval_index"


def main() -> int:
    documents = WordChunker().chunk_all(corpus_documents(CORPUS_PATH))
    provider = SentenceTransformerEmbeddingProvider()
    store = LocalVectorStore()
    store.add(documents, provider.embed([document.text for document in documents]))
    store.save(INDEX_DIRECTORY)
    print(f"Indexed {store.size} documents at {INDEX_DIRECTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
