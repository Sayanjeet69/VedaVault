"""Provider-neutral future LLM boundary; no model implementation is included."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .answer import ANSWER_CONTRACT_RULES, AnswerContract, AnswerMode
from .grounding import GROUNDING_INSTRUCTIONS, GroundingContext
from .language import LanguagePolicy


class LLMProviderError(RuntimeError):
    """A clean provider-boundary failure that callers may handle without model coupling."""


@dataclass(frozen=True)
class GenerationRequest:
    """Immutable, model-independent input for a future answer-generation provider."""

    grounding_context: GroundingContext
    mode: AnswerMode
    grounding_rules: str = GROUNDING_INSTRUCTIONS
    answer_contract_rules: str = ANSWER_CONTRACT_RULES
    generation_configuration: Mapping[str, Any] | None = None
    language_policy: LanguagePolicy | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.grounding_context, GroundingContext):
            raise ValueError("grounding_context must be a GroundingContext")
        if not isinstance(self.mode, AnswerMode):
            raise ValueError("mode must be an AnswerMode")
        if not isinstance(self.grounding_rules, str) or not self.grounding_rules.strip():
            raise ValueError("grounding_rules must be a non-empty string")
        if not isinstance(self.answer_contract_rules, str) or not self.answer_contract_rules.strip():
            raise ValueError("answer_contract_rules must be a non-empty string")
        if self.generation_configuration is not None:
            object.__setattr__(self, "generation_configuration", _freeze_mapping(self.generation_configuration))
        if self.language_policy is not None:
            if not isinstance(self.language_policy, LanguagePolicy):
                raise ValueError("language_policy must be a LanguagePolicy or None")

    @property
    def query(self) -> str:
        """Expose the original user query without duplicating it outside grounding context."""
        return self.grounding_context.query


class LLMProvider(ABC):
    """Future-provider contract, intentionally independent of any model SDK or runtime."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> AnswerContract:
        """Return a validated answer contract or raise LLMProviderError."""

    @staticmethod
    def validate_response(request: GenerationRequest, answer: AnswerContract) -> AnswerContract:
        """Ensure a provider response matches the request and supplied grounding evidence."""
        if not isinstance(answer, AnswerContract):
            raise LLMProviderError("provider returned a value that is not an AnswerContract")
        if answer.query != request.query:
            raise LLMProviderError("provider response query does not match the generation request")
        if answer.mode is not request.mode:
            raise LLMProviderError("provider response mode does not match the generation request")
        supplied_ids = frozenset(
            item.passage_id for item in request.grounding_context.evidence_items if item.passage_id is not None
        )
        if answer.evidence_passage_ids != supplied_ids:
            raise LLMProviderError("provider response evidence IDs do not match supplied grounding evidence")
        return answer


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("generation_configuration must be a mapping or None")
    return MappingProxyType({key: _freeze_value(nested_value) for key, nested_value in value.items()})


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze_value(item) for item in value)
    return value
