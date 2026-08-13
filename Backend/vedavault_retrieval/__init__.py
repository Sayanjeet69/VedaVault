"""VedaVault's corpus-agnostic local retrieval interfaces."""

from .chunking import WordChunker
from .documents import RetrievalDocument, corpus_documents, deterministic_document_id
from .embeddings import EmbeddingProvider, SentenceTransformerEmbeddingProvider
from .filters import MetadataFilter
from .retrieval import RetrievalResult, Retriever
from .vector_store import LocalVectorStore, VectorStore

__all__ = [
    "EmbeddingProvider",
    "LocalVectorStore",
    "MetadataFilter",
    "RetrievalDocument",
    "RetrievalResult",
    "Retriever",
    "SentenceTransformerEmbeddingProvider",
    "VectorStore",
    "WordChunker",
    "corpus_documents",
    "deterministic_document_id",
]
