"""Deterministic HTTP API tests; skipped when optional API dependencies are absent."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Backend"))

from vedavault_retrieval import (  # noqa: E402
    AnswerContract,
    AnswerMode,
    ClarificationRequiredError,
    EvidenceBundle,
    GroundingContext,
    LLMProviderError,
    QueryUnderstandingResult,
    RetrievalDocument,
    RetrievalResult,
    ScripturalClaim,
    SupportedLanguage,
    VedaVaultResponse,
)


API_DEPENDENCIES_AVAILABLE = all(
    importlib.util.find_spec(package) is not None
    for package in ("fastapi", "httpx")
)

if API_DEPENDENCIES_AVAILABLE:
    from fastapi.testclient import TestClient

    import vedavault_api


class FakeVedaVaultService:
    def __init__(self) -> None:
        self.calls = []

    def answer(self, query, language_policy, *, mode):
        self.calls.append((query, language_policy, mode))
        results = [
            RetrievalResult(
                RetrievalDocument(
                    "rank-1",
                    "Act steadily without attachment.",
                    {
                        "passage_id": "BG_02_47",
                        "chapter": 2,
                        "verse": 47,
                        "language": "English",
                        "text_layer": "translations",
                        "source": {"source_id": "fixture"},
                    },
                ),
                0.91,
            ),
            RetrievalResult(
                RetrievalDocument(
                    "rank-2",
                    "Remain steady in disciplined action.",
                    {
                        "passage_id": "BG_02_48",
                        "chapter": 2,
                        "verse": 48,
                        "language": "English",
                        "text_layer": "translations",
                        "source": {"source_id": "fixture"},
                    },
                ),
                0.83,
            ),
        ]
        bundle = EvidenceBundle.from_retrieval(query, results)
        context = GroundingContext.from_evidence_bundle(bundle)
        interpretation = (
            "A grounded interpretation."
            if mode is AnswerMode.PHILOSOPHICAL
            else None
        )
        application = (
            "Take the next responsible action."
            if mode is AnswerMode.APPLICATION
            else None
        )
        answer = AnswerContract.from_grounding_context(
            context,
            mode,
            (
                ScripturalClaim(
                    "The supplied verse supports disciplined action.",
                    ("BG_02_47",),
                ),
            ),
            interpretation=interpretation,
            application=application,
        )
        return VedaVaultResponse(
            original_query=query,
            retrieval_query="Bhagavad Gita teaching on disciplined action",
            language_policy=language_policy,
            evidence_bundle=bundle,
            grounding_context=context,
            answer=answer,
        )


class ClarificationService:
    def answer(self, query, language_policy, *, mode):
        raise ClarificationRequiredError(
            QueryUnderstandingResult(
                query,
                "ambiguous Bhagavad Gita question",
                language_policy,
                clarification_required=True,
            )
        )


class ProviderFailureService:
    def __init__(self, message: str) -> None:
        self.message = message

    def answer(self, query, language_policy, *, mode):
        raise LLMProviderError(self.message)


@unittest.skipUnless(
    API_DEPENDENCIES_AVAILABLE,
    "FastAPI/httpx test dependencies are optional and not installed",
)
class VedaVaultApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = FakeVedaVaultService()
        vedavault_api.app.dependency_overrides[
            vedavault_api.get_vedavault_service
        ] = lambda: self.service
        self.client = TestClient(vedavault_api.app, raise_server_exceptions=False)

    def tearDown(self) -> None:
        vedavault_api.app.dependency_overrides.clear()

    def test_health_is_deterministic_and_does_not_construct_service(self) -> None:
        with patch.object(
            vedavault_api,
            "create_vedavault_service",
            side_effect=AssertionError("health must not construct the service"),
        ):
            response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"status": "ok", "service": "vedavault", "rag_version": "v1"},
        )
        self.assertEqual(self.service.calls, [])

    def test_valid_english_textual_request(self) -> None:
        response = self.client.post(
            "/answer",
            json={"query": "What does the Gita teach?", "mode": "textual"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["response_language"], "en")
        self.assertEqual(response.json()["mode"], "textual")

    def test_hindi_application_maps_language_and_mode(self) -> None:
        response = self.client.post(
            "/answer",
            json={
                "query": "gita me kya karu",
                "input_language": "hi",
                "response_language": "hi",
                "mode": "application",
            },
        )
        self.assertEqual(response.status_code, 200)
        query, policy, mode = self.service.calls[0]
        self.assertEqual(query, "gita me kya karu")
        self.assertEqual(policy.input_languages, (SupportedLanguage.HINDI,))
        self.assertEqual(
            policy.requested_response_language,
            SupportedLanguage.HINDI,
        )
        self.assertIs(mode, AnswerMode.APPLICATION)
        self.assertEqual(response.json()["response_language"], "hi")
        self.assertEqual(
            response.json()["application"],
            "Take the next responsible action.",
        )

    def test_bengali_and_sanskrit_are_accepted(self) -> None:
        for language in ("bn", "sa"):
            with self.subTest(language=language):
                response = self.client.post(
                    "/answer",
                    json={
                        "query": "scripture question",
                        "input_language": language,
                        "response_language": language,
                    },
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["response_language"], language)

    def test_original_query_is_preserved_exactly(self) -> None:
        query = "  What should I do?  "
        response = self.client.post("/answer", json={"query": query})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["query"], query)
        self.assertEqual(self.service.calls[0][0], query)

    def test_response_includes_rewrite_and_ranked_retrieved_ids(self) -> None:
        response = self.client.post("/answer", json={"query": "question"})
        body = response.json()
        self.assertEqual(
            body["retrieval_query"],
            "Bhagavad Gita teaching on disciplined action",
        )
        self.assertEqual(body["retrieved_verse_ids"], ["BG_02_47", "BG_02_48"])

    def test_claims_and_citations_serialize_deterministically(self) -> None:
        first = self.client.post("/answer", json={"query": "question"}).json()
        second = self.client.post("/answer", json={"query": "question"}).json()
        expected = [
            {
                "statement": "The supplied verse supports disciplined action.",
                "cited_verse_ids": ["BG_02_47"],
            }
        ]
        self.assertEqual(first["scriptural_teaching"], expected)
        self.assertEqual(first["cited_verse_ids"], ["BG_02_47"])
        self.assertEqual(first, second)

    def test_invalid_language_is_rejected(self) -> None:
        response = self.client.post(
            "/answer",
            json={"query": "question", "input_language": "xx"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.service.calls, [])

    def test_non_v1_known_language_is_rejected(self) -> None:
        response = self.client.post(
            "/answer",
            json={"query": "question", "input_language": "ta"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.service.calls, [])

    def test_invalid_mode_is_rejected(self) -> None:
        response = self.client.post(
            "/answer",
            json={"query": "question", "mode": "creative"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.service.calls, [])

    def test_empty_and_whitespace_queries_are_rejected(self) -> None:
        for query in ("", "   "):
            with self.subTest(query=query):
                response = self.client.post("/answer", json={"query": query})
                self.assertEqual(response.status_code, 400)
        self.assertEqual(self.service.calls, [])

    def test_clarification_is_a_structured_conflict(self) -> None:
        vedavault_api.app.dependency_overrides[
            vedavault_api.get_vedavault_service
        ] = ClarificationService
        response = self.client.post("/answer", json={"query": "What does that mean?"})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            response.json(),
            {
                "error": "clarification_required",
                "message": "The query needs clarification before scripture can be retrieved.",
                "query": "What does that mean?",
                "clarification_required": True,
            },
        )

    def test_provider_failure_is_safe_502(self) -> None:
        vedavault_api.app.dependency_overrides[
            vedavault_api.get_vedavault_service
        ] = lambda: ProviderFailureService("private upstream failure")
        response = self.client.post("/answer", json={"query": "question"})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["error"], "upstream_service_error")
        self.assertNotIn("private upstream failure", response.text)

    def test_rate_limit_is_safe_429(self) -> None:
        secret = "offline-secret-key"
        vedavault_api.app.dependency_overrides[
            vedavault_api.get_vedavault_service
        ] = lambda: ProviderFailureService(
            f"Groq HTTP 429; organization=private; Authorization=Bearer {secret}"
        )
        response = self.client.post(
            "/answer",
            json={"query": "question"},
            headers={"Authorization": f"Bearer {secret}"},
        )
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json()["error"], "upstream_rate_limited")
        self.assertNotIn(secret, response.text)
        self.assertNotIn("Authorization", response.text)
        self.assertNotIn("organization", response.text)

    def test_service_factory_is_cached_across_requests(self) -> None:
        cached_service = object()
        vedavault_api.get_vedavault_service.cache_clear()
        try:
            with patch.object(
                vedavault_api,
                "create_vedavault_service",
                return_value=cached_service,
            ) as factory:
                first = vedavault_api.get_vedavault_service()
                second = vedavault_api.get_vedavault_service()
            self.assertIs(first, cached_service)
            self.assertIs(second, cached_service)
            factory.assert_called_once_with()
        finally:
            vedavault_api.get_vedavault_service.cache_clear()


if __name__ == "__main__":
    unittest.main()
