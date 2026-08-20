"""Model-independent query understanding and retrieval-rewrite contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .language import LanguagePolicy


QUERY_UNDERSTANDING_INSTRUCTIONS = """QUERY UNDERSTANDING RULES
- Do not answer the scripture question.
- Do not provide verse numbers.
- Do not invent or reproduce quotations.
- Return only the retrieval intent as structured JSON.
- Normalize Hinglish, Banglish, transliteration, spelling mistakes, broken grammar,
  and code-switching into one concise semantic retrieval query.
- An English retrieval query is preferred for V1 because the trusted retrieval
  corpus has complete English translations.
- Preserve recognizable Sanskrit concepts and names when they improve retrieval.
- The retrieval query is a search aid only. It is not scripture or evidence.
- Rewriting the query must not change the user-facing response language.
- Require clarification only when ambiguity materially prevents reliable retrieval.
- Return exactly one JSON object with exactly these two fields and no others:
  retrieval_query (a non-empty string) and clarification_required (a boolean).
- Do not return original_query, language_policy, explanations, verse IDs, nested
  objects, markdown, or any additional fields."""


class QueryUnderstandingProviderError(RuntimeError):
    """A clean query-understanding boundary failure."""


@dataclass(frozen=True)
class QueryUnderstandingResult:
    """Immutable separation between original user input and retrieval intent."""

    original_query: str
    retrieval_query: str
    language_policy: LanguagePolicy
    clarification_required: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.original_query, str) or not self.original_query.strip():
            raise ValueError("original_query must be a non-empty string")
        if not isinstance(self.retrieval_query, str) or not self.retrieval_query.strip():
            raise ValueError("retrieval_query must be a non-empty string")
        if not isinstance(self.language_policy, LanguagePolicy):
            raise ValueError("language_policy must be a LanguagePolicy")
        if not isinstance(self.clarification_required, bool):
            raise ValueError("clarification_required must be boolean")

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic provider-neutral representation."""
        return {
            "original_query": self.original_query,
            "retrieval_query": self.retrieval_query,
            "language_policy": self.language_policy.to_dict(),
            "clarification_required": self.clarification_required,
        }


class QueryUnderstandingProvider(ABC):
    """Provider-neutral boundary for converting user input into retrieval intent."""

    @abstractmethod
    def understand(
        self,
        original_query: str,
        language_policy: LanguagePolicy,
    ) -> QueryUnderstandingResult:
        """Return a validated retrieval rewrite without answering the query."""

    @staticmethod
    def validate_response(
        original_query: str,
        language_policy: LanguagePolicy,
        result: QueryUnderstandingResult,
    ) -> QueryUnderstandingResult:
        """Ensure a provider preserves the exact input and language-policy object."""
        if not isinstance(result, QueryUnderstandingResult):
            raise QueryUnderstandingProviderError(
                "provider returned a value that is not a QueryUnderstandingResult"
            )
        if result.original_query != original_query:
            raise QueryUnderstandingProviderError("provider changed the original query")
        if result.language_policy is not language_policy:
            raise QueryUnderstandingProviderError("provider replaced the language policy")
        return result
