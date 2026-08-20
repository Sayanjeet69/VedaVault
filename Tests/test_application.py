"""Offline tests for the provider-neutral VedaVault application service."""

from __future__ import annotations

import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Backend"))

from vedavault_retrieval import (  # noqa: E402
    AnswerContract,
    AnswerMode,
    ClarificationRequiredError,
    LLMProvider,
    LLMProviderError,
    LanguagePolicy,
    QueryUnderstandingProvider,
    QueryUnderstandingResult,
    RetrievalDocument,
    RetrievalResult,
    ScripturalClaim,
    SupportedLanguage,
    VedaVaultService,
)


class FakeQueryUnderstandingProvider(QueryUnderstandingProvider):
    def __init__(
        self,
        retrieval_query: str,
        *,
        clarification_required: bool = False,
    ) -> None:
        self.retrieval_query = retrieval_query
        self.clarification_required = clarification_required
        self.calls: list[tuple[str, LanguagePolicy]] = []

    def understand(
        self,
        original_query: str,
        language_policy: LanguagePolicy,
    ) -> QueryUnderstandingResult:
        self.calls.append((original_query, language_policy))
        return self.validate_response(
            original_query,
            language_policy,
            QueryUnderstandingResult(
                original_query,
                self.retrieval_query,
                language_policy,
                self.clarification_required,
            ),
        )


class RecordingRetriever:
    def __init__(
        self,
        results: list[RetrievalResult] | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.results = list(results or [])
        self.failure = failure
        self.calls: list[tuple[str, dict[str, object]]] = []

    def retrieve(self, query: str, **configuration):
        self.calls.append((query, configuration))
        if self.failure is not None:
            raise self.failure
        return list(self.results)


class RecordingLLMProvider(LLMProvider):
    def __init__(self) -> None:
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        if request.grounding_context.evidence_items:
            first = request.grounding_context.evidence_items[0]
            claims = (
                ScripturalClaim(
                    "The grounded evidence teaches disciplined action.",
                    (first.passage_id,),
                ),
            )
            return AnswerContract.from_grounding_context(
                request.grounding_context,
                request.mode,
                claims,
            )
        return AnswerContract.from_grounding_context(
            request.grounding_context,
            request.mode,
            evidence_sufficient=False,
            limitations=("No relevant scriptural evidence was retrieved.",),
        )


class InvalidEvidenceLLMProvider(LLMProvider):
    def generate(self, request):
        return AnswerContract(
            query=request.query,
            mode=request.mode,
            scriptural_claims=(
                ScripturalClaim("An unsupported claim.", ("BG_99_99",)),
            ),
            evidence_passage_ids=frozenset({"BG_99_99"}),
        )


class VedaVaultServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_query = "gita me result ki tension ho toh kya karu"
        self.retrieval_query = (
            "Bhagavad Gita teaching on action and non-attachment to results"
        )
        self.policy = LanguagePolicy(
            input_languages=(SupportedLanguage.HINDI, SupportedLanguage.ENGLISH),
            requested_response_language=SupportedLanguage.BENGALI,
            code_switched=True,
            transliterated=True,
        )
        self.results = [
            RetrievalResult(
                RetrievalDocument(
                    "translation-1",
                    "You have a right to action, but not to its fruits.",
                    {
                        "passage_id": "BG_02_47",
                        "chapter": 2,
                        "verse": 47,
                        "text_layer": "translations",
                        "source": {"source_id": "gita-source"},
                    },
                ),
                0.92,
            ),
            RetrievalResult(
                RetrievalDocument(
                    "translation-2",
                    "Established in yoga, perform action without attachment.",
                    {
                        "passage_id": "BG_02_48",
                        "chapter": 2,
                        "verse": 48,
                        "text_layer": "translations",
                        "source": {"source_id": "gita-source"},
                    },
                ),
                0.88,
            ),
        ]

    def make_service(self, *, results=None, clarification_required=False):
        understanding = FakeQueryUnderstandingProvider(
            self.retrieval_query,
            clarification_required=clarification_required,
        )
        retriever = RecordingRetriever(self.results if results is None else results)
        generator = RecordingLLMProvider()
        service = VedaVaultService(understanding, retriever, generator)  # type: ignore[arg-type]
        return service, understanding, retriever, generator

    def test_successful_pipeline_preserves_queries_policy_and_retrieval_defaults(self) -> None:
        service, understanding, retriever, generator = self.make_service()
        response = service.answer(
            self.original_query,
            self.policy,
            mode=AnswerMode.TEXTUAL,
        )

        self.assertEqual(understanding.calls, [(self.original_query, self.policy)])
        self.assertEqual(retriever.calls[0][0], self.retrieval_query)
        self.assertNotEqual(retriever.calls[0][0], self.original_query)
        self.assertEqual(
            retriever.calls[0][1],
            {
                "limit": 5,
                "text_layers": ("translations",),
                "deduplicate_by_verse": True,
                "diversity_candidate_limit": 100,
            },
        )
        self.assertEqual(response.original_query, self.original_query)
        self.assertEqual(response.retrieval_query, self.retrieval_query)
        self.assertIs(response.language_policy, self.policy)
        self.assertEqual(
            response.language_policy.effective_primary_response_language,
            SupportedLanguage.BENGALI,
        )
        self.assertEqual(response.retrieved_passage_ids, ("BG_02_47", "BG_02_48"))
        self.assertIs(
            response.grounding_context,
            generator.requests[0].grounding_context,
        )
        self.assertEqual(response.answer.query, self.original_query)
        self.assertEqual(len(generator.requests), 1)
        with self.assertRaises(FrozenInstanceError):
            response.original_query = "changed"  # type: ignore[misc]

    def test_generation_receives_original_query_and_ranked_grounded_evidence(self) -> None:
        service, _, _, generator = self.make_service()
        response = service.answer(
            self.original_query,
            self.policy,
            mode=AnswerMode.TEXTUAL,
        )
        request = generator.requests[0]
        context = request.grounding_context
        self.assertEqual(request.query, self.original_query)
        self.assertIs(request.language_policy, self.policy)
        self.assertEqual(
            [item.passage_id for item in context.evidence_items],
            ["BG_02_47", "BG_02_48"],
        )
        self.assertEqual(context.evidence_items[0].source["source_id"], "gita-source")
        self.assertEqual(context.evidence_items[0].text_layer, "translations")
        self.assertNotIn(
            self.retrieval_query,
            context.to_prompt_context(),
        )
        self.assertEqual(
            response.answer.evidence_passage_ids,
            frozenset({"BG_02_47", "BG_02_48"}),
        )

    def test_clarification_stops_before_retrieval_and_generation(self) -> None:
        service, _, retriever, generator = self.make_service(
            clarification_required=True
        )
        with self.assertRaises(ClarificationRequiredError) as raised:
            service.answer(
                self.original_query,
                self.policy,
                mode=AnswerMode.TEXTUAL,
            )
        self.assertEqual(retriever.calls, [])
        self.assertEqual(generator.requests, [])
        self.assertEqual(
            raised.exception.understanding.original_query,
            self.original_query,
        )
        self.assertIs(raised.exception.understanding.language_policy, self.policy)

    def test_retrieval_failure_propagates_without_generation(self) -> None:
        failure = RuntimeError("local retrieval unavailable")
        understanding = FakeQueryUnderstandingProvider(self.retrieval_query)
        retriever = RecordingRetriever(failure=failure)
        generator = RecordingLLMProvider()
        service = VedaVaultService(understanding, retriever, generator)  # type: ignore[arg-type]
        with self.assertRaises(RuntimeError) as raised:
            service.answer(
                self.original_query,
                self.policy,
                mode=AnswerMode.TEXTUAL,
            )
        self.assertIs(raised.exception, failure)
        self.assertEqual(generator.requests, [])

    def test_empty_retrieval_is_grounded_and_returns_insufficient_evidence(self) -> None:
        service, _, _, generator = self.make_service(results=[])
        response = service.answer(
            self.original_query,
            self.policy,
            mode=AnswerMode.TEXTUAL,
        )
        self.assertEqual(response.evidence_bundle.items, ())
        self.assertEqual(generator.requests[0].grounding_context.evidence_items, ())
        self.assertFalse(response.answer.evidence_sufficient)
        self.assertEqual(response.answer.evidence_passage_ids, frozenset())
        self.assertIn("No relevant scriptural evidence", response.answer.limitations[0])

    def test_service_revalidates_provider_answer_against_grounded_evidence(self) -> None:
        understanding = FakeQueryUnderstandingProvider(self.retrieval_query)
        retriever = RecordingRetriever(self.results)
        service = VedaVaultService(
            understanding,
            retriever,  # type: ignore[arg-type]
            InvalidEvidenceLLMProvider(),
        )
        with self.assertRaisesRegex(
            LLMProviderError,
            "evidence IDs do not match supplied grounding evidence",
        ):
            service.answer(
                self.original_query,
                self.policy,
                mode=AnswerMode.TEXTUAL,
            )


if __name__ == "__main__":
    unittest.main()
