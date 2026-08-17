"""Embedding provider contract and the local Sentence Transformers adapter."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EmbeddingWorkload:
    """Counts useful for sizing an embedding run without running inference."""

    input_count: int
    unique_input_count: int
    batch_size: int

    @property
    def batch_count(self) -> int:
        return (self.unique_input_count + self.batch_size - 1) // self.batch_size


@dataclass(frozen=True)
class EmbeddingPromptProfile:
    """Provider-local document/query prompting for a specific embedding model."""

    name: str
    document_prefix: str = ""
    query_prefix: str = ""


E5_PROMPT_PROFILE = EmbeddingPromptProfile("e5", document_prefix="passage: ", query_prefix="query: ")
NO_PROMPT_PROFILE = EmbeddingPromptProfile("none")


@dataclass(frozen=True)
class EmbeddingConfiguration:
    """Versioned, device-independent settings that determine vector semantics."""

    model_name: str
    embedding_dimension: int
    prompt_profile: str
    document_prefix: str
    query_prefix: str
    max_seq_length: int | None
    normalize_embeddings: bool = True
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "prompt_profile": self.prompt_profile,
            "document_prefix": self.document_prefix,
            "query_prefix": self.query_prefix,
            "max_seq_length": self.max_seq_length,
            "normalize_embeddings": self.normalize_embeddings,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EmbeddingConfiguration":
        required = {
            "schema_version", "model_name", "embedding_dimension", "prompt_profile",
            "document_prefix", "query_prefix", "max_seq_length", "normalize_embeddings",
        }
        if set(value) != required or value["schema_version"] != 1:
            raise ValueError("index manifest has an unsupported embedding configuration")
        if not isinstance(value["model_name"], str) or not value["model_name"]:
            raise ValueError("index manifest embedding model_name must be a non-empty string")
        if not isinstance(value["embedding_dimension"], int) or value["embedding_dimension"] < 1:
            raise ValueError("index manifest embedding_dimension must be a positive integer")
        if not all(isinstance(value[key], str) for key in ("prompt_profile", "document_prefix", "query_prefix")):
            raise ValueError("index manifest prompt configuration must contain strings")
        if value["max_seq_length"] is not None and (
            not isinstance(value["max_seq_length"], int) or value["max_seq_length"] < 1
        ):
            raise ValueError("index manifest max_seq_length must be null or a positive integer")
        if not isinstance(value["normalize_embeddings"], bool):
            raise ValueError("index manifest normalize_embeddings must be boolean")
        return cls(**value)


class EmbeddingProvider(ABC):
    """An interchangeable text-to-vector provider."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return one float vector per input text."""

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed indexable documents; providers may specialize this operation."""
        return self.embed(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a user query; providers may specialize this operation."""
        return self.embed([query])

    def index_configuration(self, embedding_dimension: int) -> EmbeddingConfiguration | None:
        """Return persisted semantic settings, if this provider supports them."""
        return None


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Lazy local adapter; importing this package does not load a model."""

    DEFAULT_MODEL = "intfloat/multilingual-e5-small"
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_CPU_THREADS = min(8, os.cpu_count() or 1)

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        cpu_threads: int | None = None,
        max_seq_length: int | None = None,
        prompt_profile: EmbeddingPromptProfile | None = None,
        local_files_only: bool = False,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name or os.getenv("VEDAVAULT_EMBEDDING_MODEL", self.DEFAULT_MODEL)
        self.device = device or os.getenv("VEDAVAULT_EMBEDDING_DEVICE", "cpu")
        self.batch_size = batch_size if batch_size is not None else _positive_env_int(
            "VEDAVAULT_EMBEDDING_BATCH_SIZE", self.DEFAULT_BATCH_SIZE
        )
        self.cpu_threads = cpu_threads if cpu_threads is not None else _positive_env_int(
            "VEDAVAULT_EMBEDDING_CPU_THREADS", self.DEFAULT_CPU_THREADS
        )
        self.max_seq_length = max_seq_length if max_seq_length is not None else _positive_env_int_or_none(
            "VEDAVAULT_EMBEDDING_MAX_SEQ_LENGTH"
        )
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")
        self.prompt_profile = prompt_profile or _prompt_profile_from_env()
        self.local_files_only = local_files_only
        self._model = model

    def _get_model(self):
        if self._model is None:
            try:
                import torch
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for local semantic retrieval. "
                    "Install Backend/requirements-retrieval.txt."
                ) from exc
            if self.device.lower() == "cpu":
                torch.set_num_threads(self.cpu_threads)
            self._model = SentenceTransformer(
                self.model_name, device=self.device, local_files_only=self.local_files_only
            )
        if self.max_seq_length is not None:
            self._model.max_seq_length = self.max_seq_length
        return self._model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed text in deterministic, bounded batches without role prefixes."""
        return self._embed(texts)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed documents with this provider's explicitly selected prompt profile."""
        return self._embed([f"{self.prompt_profile.document_prefix}{text}" for text in texts])

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query with this provider's explicitly selected prompt profile."""
        return self._embed([f"{self.prompt_profile.query_prefix}{query}"])

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        # Preserve requested result order while avoiding duplicate model work.
        values = list(dict.fromkeys(texts))
        model = self._get_model()
        vectors = np.asarray(
            model.encode(
                values,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        vector_by_text = dict(zip(values, vectors, strict=True))
        return np.asarray([vector_by_text[text] for text in texts], dtype=np.float32)

    def document_workload(self, texts: Sequence[str]) -> EmbeddingWorkload:
        """Report document embedding work after exact-text reuse."""
        return EmbeddingWorkload(len(texts), len(set(texts)), self.batch_size)

    def index_configuration(self, embedding_dimension: int) -> EmbeddingConfiguration:
        if embedding_dimension < 1:
            raise ValueError("embedding_dimension must be positive")
        return EmbeddingConfiguration(
            model_name=self.model_name,
            embedding_dimension=embedding_dimension,
            prompt_profile=self.prompt_profile.name,
            document_prefix=self.prompt_profile.document_prefix,
            query_prefix=self.prompt_profile.query_prefix,
            max_seq_length=self.max_seq_length,
        )


def _positive_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _positive_env_int_or_none(name: str) -> int | None:
    value = os.getenv(name)
    if value is None:
        return None
    return _positive_env_int(name, 0)


def _prompt_profile_from_env() -> EmbeddingPromptProfile:
    """Select a built-in profile explicitly; custom profiles are passed to the constructor."""
    configured_name = os.getenv("VEDAVAULT_EMBEDDING_PROMPT_PROFILE", E5_PROMPT_PROFILE.name)
    profiles = {E5_PROMPT_PROFILE.name: E5_PROMPT_PROFILE, NO_PROMPT_PROFILE.name: NO_PROMPT_PROFILE}
    try:
        return profiles[configured_name]
    except KeyError as exc:
        raise ValueError(
            "VEDAVAULT_EMBEDDING_PROMPT_PROFILE must be one of: " + ", ".join(sorted(profiles))
        ) from exc
