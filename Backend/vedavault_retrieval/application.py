"""Provider-neutral orchestration for one complete VedaVault query."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .answer import AnswerContract, AnswerMode
from .conversation import ConversationContext
from .evidence import EvidenceBundle
from .evidence_hygiene import EvidenceHygienePolicy
from .grounding import GroundingContext
from .language import LanguagePolicy
from .llm import GenerationRequest, LLMProvider
from .query_understanding import (
    QueryUnderstandingProvider,
    QueryUnderstandingResult,
)
from .retrieval import Retriever


V1_CONTEXT_LIMIT = 5
V1_DIVERSITY_CANDIDATE_LIMIT = 100
V1_TEXT_LAYERS = ("translations",)


class ClarificationRequiredError(RuntimeError):
    """Application state indicating that retrieval must wait for clarification."""

    def __init__(self, understanding: QueryUnderstandingResult) -> None:
        if not isinstance(understanding, QueryUnderstandingResult):
            raise ValueError("understanding must be a QueryUnderstandingResult")
        self.understanding = understanding
        super().__init__("query understanding requires clarification")


@dataclass(frozen=True)
class VedaVaultResponse:
    """Immutable grounded response with ranked evidence retained for traceability."""

    original_query: str
    retrieval_query: str
    language_policy: LanguagePolicy
    evidence_bundle: EvidenceBundle
    grounding_context: GroundingContext
    answer: AnswerContract

    def __post_init__(self) -> None:
        if not isinstance(self.original_query, str) or not self.original_query.strip():
            raise ValueError("original_query must be a non-empty string")
        if not isinstance(self.retrieval_query, str) or not self.retrieval_query.strip():
            raise ValueError("retrieval_query must be a non-empty string")
        if not isinstance(self.language_policy, LanguagePolicy):
            raise ValueError("language_policy must be a LanguagePolicy")
        if not isinstance(self.evidence_bundle, EvidenceBundle):
            raise ValueError("evidence_bundle must be an EvidenceBundle")
        if not isinstance(self.grounding_context, GroundingContext):
            raise ValueError("grounding_context must be a GroundingContext")
        if not isinstance(self.answer, AnswerContract):
            raise ValueError("answer must be an AnswerContract")
        if self.evidence_bundle.query != self.original_query:
            raise ValueError("evidence bundle must preserve the original query")
        if self.grounding_context.query != self.original_query:
            raise ValueError("grounding context must preserve the original query")
        if self.answer.query != self.original_query:
            raise ValueError("answer must preserve the original query")

    @property
    def retrieved_passage_ids(self) -> tuple[str, ...]:
        """Return retrieved canonical verse IDs in grounding rank order."""
        return tuple(
            item.passage_id
            for item in self.evidence_bundle.items
            if item.passage_id is not None
        )


class VedaVaultService:
    """Compose provider-neutral understanding, retrieval, grounding, and generation."""

    def __init__(
        self,
        query_understanding_provider: QueryUnderstandingProvider,
        retriever: Retriever,
        llm_provider: LLMProvider,
        *,
        context_limit: int = V1_CONTEXT_LIMIT,
        diversity_candidate_limit: int = V1_DIVERSITY_CANDIDATE_LIMIT,
        evidence_hygiene_policy: EvidenceHygienePolicy | None = None,
    ) -> None:
        if not isinstance(context_limit, int) or isinstance(context_limit, bool) or context_limit < 1:
            raise ValueError("context_limit must be a positive integer")
        if (
            not isinstance(diversity_candidate_limit, int)
            or isinstance(diversity_candidate_limit, bool)
            or diversity_candidate_limit < context_limit
        ):
            raise ValueError(
                "diversity_candidate_limit must be an integer at least as large as context_limit"
            )
        self.query_understanding_provider = query_understanding_provider
        self.retriever = retriever
        self.llm_provider = llm_provider
        self.context_limit = context_limit
        self.diversity_candidate_limit = diversity_candidate_limit
        self.evidence_hygiene_policy = evidence_hygiene_policy

    def answer(
        self,
        query: str,
        language_policy: LanguagePolicy,
        *,
        mode: AnswerMode,
        generation_configuration: Mapping[str, Any] | None = None,
        conversation_context: ConversationContext | None = None,
    ) -> VedaVaultResponse:
        """Run one query through the frozen V1 pipeline."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(language_policy, LanguagePolicy):
            raise ValueError("language_policy must be a LanguagePolicy")
        if not isinstance(mode, AnswerMode):
            raise ValueError("mode must be an AnswerMode")
        if conversation_context is not None and not isinstance(
            conversation_context, ConversationContext
        ):
            raise ValueError("conversation_context must be a ConversationContext or None")

        if conversation_context is None:
            understanding = self.query_understanding_provider.understand(
                query,
                language_policy,
            )
        else:
            understanding = self.query_understanding_provider.understand(
                query,
                language_policy,
                conversation_context=conversation_context,
            )
        understanding = QueryUnderstandingProvider.validate_response(
            query,
            language_policy,
            understanding,
        )
        if understanding.clarification_required:
            raise ClarificationRequiredError(understanding)

        retrieval_results = self.retriever.retrieve(
            understanding.retrieval_query,
            limit=self.context_limit,
            text_layers=V1_TEXT_LAYERS,
            deduplicate_by_verse=True,
            diversity_candidate_limit=self.diversity_candidate_limit,
        )
        evidence_bundle = EvidenceBundle.from_retrieval(
            query,
            retrieval_results,
            retrieval_configuration={
                "limit": self.context_limit,
                "text_layers": V1_TEXT_LAYERS,
                "deduplicate_by_verse": True,
                "diversity_candidate_limit": self.diversity_candidate_limit,
            },
        )
        grounding_context = GroundingContext.from_evidence_bundle(
            evidence_bundle,
            self.evidence_hygiene_policy,
        )
        generation_request = GenerationRequest(
            grounding_context,
            mode,
            generation_configuration=generation_configuration,
            language_policy=understanding.language_policy,
        )
        answer = self.llm_provider.generate(generation_request)
        answer = LLMProvider.validate_response(generation_request, answer)
        return VedaVaultResponse(
            original_query=query,
            retrieval_query=understanding.retrieval_query,
            language_policy=understanding.language_policy,
            evidence_bundle=evidence_bundle,
            grounding_context=grounding_context,
            answer=answer,
        )
