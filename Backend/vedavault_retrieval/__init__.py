"""VedaVault's corpus-agnostic local retrieval interfaces."""

from .chunking import WordChunker
from .answer import ANSWER_CONTRACT_RULES, AnswerContract, AnswerMode, ScripturalClaim
from .documents import RetrievalDocument, corpus_documents, deterministic_document_id
from .evidence import EvidenceBundle, EvidenceItem
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
from .grounding import GROUNDING_INSTRUCTIONS, GroundingContext
from .language import SUPPORTED_LANGUAGES, LanguagePolicy, SupportedLanguage, WritingScript
from .llm import GenerationRequest, LLMProvider, LLMProviderError
from .evaluation import (
    BENCHMARK_VERSION,
    BREAKDOWN_DIMENSIONS,
    CATEGORIES,
    DIFFICULTY_TAGS,
    AggregateRetrievalEvaluation,
    EvaluationQuestion,
    RetrievalBenchmark,
    RetrievalEvaluation,
    aggregate_evaluations,
    evaluate_results,
    load_evaluation_questions,
    load_retrieval_benchmark,
    validate_benchmark_corpus_references,
)
from .retrieval import RetrievalResult, Retriever, deduplicate_by_passage
from .vector_store import IndexCompatibilityError, IndexManifest, IndexManifestError, LocalVectorStore, VectorStore

__all__ = [
    "E5_PROMPT_PROFILE",
    "ANSWER_CONTRACT_RULES",
    "AggregateRetrievalEvaluation",
    "AnswerContract",
    "AnswerMode",
    "BENCHMARK_VERSION",
    "BREAKDOWN_DIMENSIONS",
    "CATEGORIES",
    "DIFFICULTY_TAGS",
    "EmbeddingConfiguration",
    "EmbeddingProvider",
    "EmbeddingPromptProfile",
    "EmbeddingWorkload",
    "EvidenceBundle",
    "EvidenceItem",
    "EvaluationQuestion",
    "GROUNDING_INSTRUCTIONS",
    "GroundingContext",
    "GenerationRequest",
    "IndexCompatibilityError",
    "IndexManifest",
    "IndexManifestError",
    "LocalVectorStore",
    "LanguagePolicy",
    "LLMProvider",
    "LLMProviderError",
    "MetadataFilter",
    "NO_PROMPT_PROFILE",
    "RetrievalDocument",
    "RetrievalBenchmark",
    "RetrievalEvaluation",
    "RetrievalResult",
    "Retriever",
    "SentenceTransformerEmbeddingProvider",
    "ScripturalClaim",
    "SUPPORTED_LANGUAGES",
    "SupportedLanguage",
    "VectorStore",
    "WordChunker",
    "WritingScript",
    "aggregate_evaluations",
    "corpus_documents",
    "deterministic_document_id",
    "deduplicate_by_passage",
    "evaluate_results",
    "load_evaluation_questions",
    "load_retrieval_benchmark",
    "validate_benchmark_corpus_references",
]
