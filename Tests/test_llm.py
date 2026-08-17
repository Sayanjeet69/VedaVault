"""Offline tests for the provider-neutral future LLM boundary."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Backend"))

from vedavault_retrieval import (  # noqa: E402
    AnswerContract,
    AnswerMode,
    EvidenceBundle,
    GenerationRequest,
    GroundingContext,
    LLMProvider,
    LLMProviderError,
    RetrievalDocument,
    RetrievalResult,
    ScripturalClaim,
)


class DeterministicFakeProvider(LLMProvider):
    """Test double that returns a supplied contract; it performs no generation."""

    def __init__(self, answer: AnswerContract) -> None:
        self.answer = answer
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> AnswerContract:
        self.requests.append(request)
        return self.validate_response(request, self.answer)


class FailingFakeProvider(LLMProvider):
    def generate(self, request: GenerationRequest) -> AnswerContract:
        raise LLMProviderError("provider unavailable in this environment")


class LLMProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        result = RetrievalResult(
            RetrievalDocument(
                "first",
                "Act without attachment.",
                {"passage_id": "BG_02_47", "chapter": 2, "verse": 47, "text_layer": "translations", "source": {"source_id": "gita-source"}},
            ),
            0.91,
        )
        self.context = GroundingContext.from_evidence_bundle(EvidenceBundle.from_retrieval("How should I act?", [result]))
        self.claim = ScripturalClaim("The evidence teaches action without attachment.", ("BG_02_47",))
        self.answer = AnswerContract.from_grounding_context(self.context, AnswerMode.TEXTUAL, (self.claim,))

    def test_request_preserves_query_context_mode_rules_and_configuration(self) -> None:
        configuration = {"output_language": "English", "audience": ["general"]}
        request = GenerationRequest(self.context, AnswerMode.TEXTUAL, generation_configuration=configuration)
        configuration["audience"].append("changed")
        self.assertEqual(request.query, "How should I act?")
        self.assertIs(request.grounding_context, self.context)
        self.assertEqual(request.mode, AnswerMode.TEXTUAL)
        self.assertIn("GROUNDING RULES", request.grounding_rules)
        self.assertIn("ANSWER CONTRACT RULES", request.answer_contract_rules)
        self.assertEqual(request.generation_configuration["audience"], ("general",))
        with self.assertRaises(TypeError):
            request.generation_configuration["output_language"] = "Sanskrit"  # type: ignore[index]

    def test_fake_provider_returns_a_validated_contract_deterministically(self) -> None:
        request = GenerationRequest(self.context, AnswerMode.TEXTUAL)
        provider = DeterministicFakeProvider(self.answer)
        self.assertEqual(provider.generate(request), self.answer)
        self.assertEqual(provider.generate(request), self.answer)
        self.assertEqual(provider.requests, [request, request])

    def test_provider_interface_and_failures_are_clean(self) -> None:
        with self.assertRaises(TypeError):
            LLMProvider()
        with self.assertRaises(LLMProviderError):
            FailingFakeProvider().generate(GenerationRequest(self.context, AnswerMode.TEXTUAL))

    def test_response_validation_rejects_mismatched_contracts(self) -> None:
        request = GenerationRequest(self.context, AnswerMode.TEXTUAL)
        wrong_query = AnswerContract(
            "another question", AnswerMode.TEXTUAL, (self.claim,), evidence_passage_ids=frozenset({"BG_02_47"})
        )
        wrong_mode = AnswerContract.from_grounding_context(self.context, AnswerMode.PHILOSOPHICAL, (self.claim,), interpretation="synthesis")
        with self.assertRaises(LLMProviderError):
            LLMProvider.validate_response(request, wrong_query)
        with self.assertRaises(LLMProviderError):
            LLMProvider.validate_response(request, wrong_mode)

    def test_invalid_request_configuration_is_rejected_without_model_dependencies(self) -> None:
        with self.assertRaises(ValueError):
            GenerationRequest(self.context, "textual")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            GenerationRequest(self.context, AnswerMode.TEXTUAL, generation_configuration=[("not", "a mapping")])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
