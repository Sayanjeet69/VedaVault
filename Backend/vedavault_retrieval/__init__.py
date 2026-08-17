"""VedaVault's corpus-agnostic local retrieval interfaces."""

from .chunking import WordChunker
from .documents import RetrievalDocument, corpus_documents, deterministic_document_id
from .embeddings import (
    E5_PROMPT_PROFILE,
    NO_PROMPT_PROFILE,
    EmbeddingConfiguration,
    EmbeddingProvider,
    EmbeddingPromptProfile,
    EmbeddingWorkload,
    SentenceTransformerEmbeddingProvider,
)
from .filters import MetadataFilter
from .retrieval import RetrievalResult, Retriever
from .vector_store import IndexCompatibilityError, IndexManifest, IndexManifestError, LocalVectorStore, VectorStore

__all__ = [
    "E5_PROMPT_PROFILE",
    "EmbeddingConfiguration",
    "EmbeddingProvider",
    "EmbeddingPromptProfile",
    "EmbeddingWorkload",
    "IndexCompatibilityError",
    "IndexManifest",
    "IndexManifestError",
    "LocalVectorStore",
    "MetadataFilter",
    "NO_PROMPT_PROFILE",
    "RetrievalDocument",
    "RetrievalResult",
    "Retriever",
    "SentenceTransformerEmbeddingProvider",
    "VectorStore",
    "WordChunker",
    "corpus_documents",
    "deterministic_document_id",
]
