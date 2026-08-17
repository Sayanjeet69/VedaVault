"""Build the first local semantic index from the canonical Bhagavad Gita corpus."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Backend"))

from vedavault_retrieval import (  # noqa: E402
    IndexManifest,
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
    workload = provider.document_workload([document.text for document in documents])
    print(
        "Embedding "
        f"{workload.input_count} index documents ({workload.unique_input_count} unique texts) "
        f"in {workload.batch_count} batches of up to {workload.batch_size}."
    )
    store = LocalVectorStore()
    vectors = provider.embed_documents([document.text for document in documents])
    store.add(documents, vectors)
    store.set_manifest(IndexManifest(provider.index_configuration(vectors.shape[1])))
    store.save(INDEX_DIRECTORY)
    print(f"Indexed {store.size} documents at {INDEX_DIRECTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
