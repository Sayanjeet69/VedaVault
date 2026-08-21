"""Offline tests for Groq-backed query understanding and generation."""

from __future__ import annotations

import io
import json
import os
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Backend"))

from vedavault_retrieval import (  # noqa: E402
    DEFAULT_GROQ_ENDPOINT,
    DEFAULT_GROQ_MODEL,
    AnswerMode,
    ConversationContext,
    ConversationRole,
    ConversationTurn,
    EvidenceBundle,
    GenerationRequest,
    GroqClient,
    GroqLLMProvider,
    GroqQueryUnderstandingProvider,
    GroundingContext,
    LLMProviderError,
    LanguagePolicy,
    QueryUnderstandingProviderError,
    RetrievalDocument,
    RetrievalResult,
    SupportedLanguage,
)


class FakeTransport:
    """Queue-backed transport that records requests and never touches the network."""

    def __init__(self, *responses: bytes | Exception) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def post(self, url, headers, body, timeout) -> bytes:
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": json.loads(body.decode("utf-8")),
                "timeout": timeout,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeHTTPResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def api_response(model_content: str) -> bytes:
    return json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": model_content}}]}
    ).encode("utf-8")


def answer_content(
    *,
    query: str = "How should I act?",
    cited_ids: list[str] | None = None,
    evidence_ids: list[str] | None = None,
    mode: str = "textual",
    interpretation: str | None = None,
    application: str | None = None,
    evidence_sufficient: bool = True,
    limitations: list[str] | None = None,
) -> str:
    cited_ids = ["BG_02_47"] if cited_ids is None else cited_ids
    evidence_ids = ["BG_02_47"] if evidence_ids is None else evidence_ids
    teaching = (
        []
        if not cited_ids
        else [
            {
                "statement": "The evidence teaches action without attachment.",
                "cited_verse_ids": cited_ids,
            }
        ]
    )
    return json.dumps(
        {
            "query": query,
            "mode": mode,
            "scriptural_teaching": teaching,
            "interpretation": interpretation,
            "application": application,
            "evidence_sufficient": evidence_sufficient,
            "limitations": limitations or [],
            "evidence_passage_ids": evidence_ids,
        }
    )


class GroqProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        result = RetrievalResult(
            RetrievalDocument(
                "translation-1",
                "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन।\nAct without attachment to results.",
                {
                    "passage_id": "BG_02_47",
                    "chapter": 2,
                    "verse": 47,
                    "text_layer": "translations",
                    "source": {"source_id": "gita-source"},
                },
            ),
            0.91,
        )
        self.context = GroundingContext.from_evidence_bundle(
            EvidenceBundle.from_retrieval("How should I act?", [result])
        )
        self.policy = LanguagePolicy(
            input_languages=(SupportedLanguage.HINDI,),
            requested_response_language=SupportedLanguage.ENGLISH,
        )
        self.request = GenerationRequest(
            self.context,
            AnswerMode.TEXTUAL,
            language_policy=self.policy,
        )

    def test_missing_environment_key_fails_cleanly_without_transport_call(self) -> None:
        transport = FakeTransport()
        with patch.dict(os.environ, {}, clear=True):
            provider = GroqLLMProvider(client=GroqClient(transport=transport))
            with self.assertRaisesRegex(LLMProviderError, "GROQ_API_KEY"):
                provider.generate(self.request)
        self.assertEqual(transport.calls, [])

    def test_successful_generation_uses_grounding_and_deterministic_configuration(self) -> None:
        transport = FakeTransport(api_response(answer_content()))
        with patch.dict(os.environ, {"GROQ_API_KEY": "offline-test-key"}, clear=True):
            provider = GroqLLMProvider(client=GroqClient(transport=transport))
            answer = provider.generate(self.request)

        self.assertEqual(answer.query, self.request.query)
        self.assertEqual(answer.cited_verse_ids, ("BG_02_47",))
        self.assertEqual(answer.evidence_passage_ids, frozenset({"BG_02_47"}))
        call = transport.calls[0]
        payload = call["payload"]
        self.assertEqual(call["url"], DEFAULT_GROQ_ENDPOINT)
        self.assertEqual(call["timeout"], 30.0)
        self.assertEqual(call["headers"]["Authorization"], "Bearer offline-test-key")
        self.assertEqual(call["headers"]["Content-Type"], "application/json")
        self.assertEqual(call["headers"]["Accept"], "application/json")
        self.assertEqual(call["headers"]["User-Agent"], "VedaVault/1.0 Python")
        self.assertEqual(payload["model"], DEFAULT_GROQ_MODEL)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["max_completion_tokens"], 1400)
        self.assertFalse(payload["stream"])
        system_prompt = payload["messages"][0]["content"]
        evidence_prompt = payload["messages"][1]["content"]
        self.assertIn("Use only the supplied scriptural evidence", system_prompt)
        self.assertIn("Preserve supplied Sanskrit exactly", system_prompt)
        self.assertIn("Never represent generated explanation", system_prompt)
        self.assertIn('"requested_response_language": "en"', system_prompt)
        self.assertIn('"effective_primary_response_language": "en"', system_prompt)
        self.assertIn("RESOLVED RESPONSE LANGUAGE\nen", system_prompt)
        self.assertIn("Write generated prose in natural English", system_prompt)
        self.assertIn("BG_02_47", system_prompt)
        self.assertIn("कर्मण्येवाधिकारस्ते", evidence_prompt)

    def test_response_language_instructions_require_resolved_native_scripts(self) -> None:
        cases = (
            (
                LanguagePolicy(input_languages=(SupportedLanguage.HINDI,)),
                "RESOLVED RESPONSE LANGUAGE\nhi",
                "natural Hindi using Devanagari script",
                "Hinglish input does not authorize Romanized output",
            ),
            (
                LanguagePolicy(input_languages=(SupportedLanguage.BENGALI,)),
                "RESOLVED RESPONSE LANGUAGE\nbn",
                "natural Bengali using Bengali script",
                "Banglish input does not authorize Romanized output",
            ),
            (
                LanguagePolicy(input_languages=(SupportedLanguage.ENGLISH,)),
                "RESOLVED RESPONSE LANGUAGE\nen",
                "natural English",
                None,
            ),
            (
                LanguagePolicy(input_languages=(SupportedLanguage.SANSKRIT,)),
                "RESOLVED RESPONSE LANGUAGE\nsa",
                "Sanskrit using Devanagari script",
                None,
            ),
        )
        for policy, resolved, instruction, extra in cases:
            with self.subTest(language=policy.effective_primary_response_language):
                transport = FakeTransport(api_response(answer_content()))
                provider = GroqLLMProvider(
                    client=GroqClient(
                        api_key="offline-test-key",
                        transport=transport,
                    )
                )
                provider.generate(
                    GenerationRequest(
                        self.context,
                        AnswerMode.TEXTUAL,
                        language_policy=policy,
                    )
                )
                prompt = transport.calls[0]["payload"]["messages"][0]["content"]
                self.assertIn(resolved, prompt)
                self.assertIn(instruction, prompt)
                if extra is not None:
                    self.assertIn(extra, prompt)

    def test_explicit_response_language_override_controls_script_instruction(self) -> None:
        policy = LanguagePolicy(
            input_languages=(SupportedLanguage.HINDI,),
            requested_response_language=SupportedLanguage.BENGALI,
            transliterated=True,
        )
        transport = FakeTransport(api_response(answer_content()))
        provider = GroqLLMProvider(
            client=GroqClient(api_key="offline-test-key", transport=transport)
        )
        provider.generate(
            GenerationRequest(
                self.context,
                AnswerMode.TEXTUAL,
                language_policy=policy,
            )
        )
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn("RESOLVED RESPONSE LANGUAGE\nbn", prompt)
        self.assertIn("natural Bengali using Bengali script", prompt)
        self.assertNotIn("natural Hindi using Devanagari script", prompt)
        self.assertIn("explicit requested response-language override", prompt)

    def test_prompt_requires_selective_smallest_sufficient_citations(self) -> None:
        transport = FakeTransport(api_response(answer_content()))
        provider = GroqLLMProvider(
            client=GroqClient(api_key="offline-test-key", transport=transport)
        )
        provider.generate(self.request)
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        normalized = " ".join(prompt.split())
        self.assertIn("candidate evidence, not mandatory citations", normalized)
        self.assertIn("smallest sufficient citation set", normalized)
        self.assertIn("Weaker or secondary passages may be omitted", normalized)
        self.assertIn("does not mean every supplied passage must be cited", normalized)
        self.assertIn("every scriptural teaching must have at least one", normalized)

    def test_prompt_forbids_memory_doctrine_and_requires_textual_support(self) -> None:
        transport = FakeTransport(api_response(answer_content()))
        provider = GroqLLMProvider(
            client=GroqClient(api_key="offline-test-key", transport=transport)
        )
        provider.generate(self.request)
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        normalized = " ".join(prompt.split())
        self.assertIn("SCRIPTURE CLAIMS ARE EVIDENCE-ONLY", normalized)
        self.assertIn("faithful paraphrase of the text in its cited supplied evidence", normalized)
        self.assertIn("Never introduce Bhagavad Gita doctrine from pretrained memory", normalized)
        self.assertIn("If you know a relevant Gita teaching", normalized)
        self.assertIn("absent from the supplied evidence, do not use it", normalized)
        self.assertIn("text directly supports that statement", normalized)

    def test_prompt_prevents_unsupported_doctrine_in_semantic_layers(self) -> None:
        transport = FakeTransport(api_response(answer_content()))
        provider = GroqLLMProvider(
            client=GroqClient(api_key="offline-test-key", transport=transport)
        )
        provider.generate(self.request)
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        normalized = " ".join(prompt.split())
        self.assertIn("must not smuggle in a new", normalized)
        self.assertIn("supported implication is allowed", normalized)
        self.assertIn("new doctrine from model memory is forbidden", normalized)
        self.assertIn("commentary as direct canonical scripture", normalized)

    def test_prompt_explicitly_allows_ignoring_irrelevant_candidates(self) -> None:
        transport = FakeTransport(api_response(answer_content()))
        provider = GroqLLMProvider(
            client=GroqClient(api_key="offline-test-key", transport=transport)
        )
        provider.generate(self.request)
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        normalized = " ".join(prompt.split())
        self.assertIn("Irrelevant or weak passages may be ignored", normalized)
        self.assertIn("smallest sufficient citation set and evidence subset", normalized)
        self.assertIn("Citing one supplied verse out of five is acceptable", normalized)
        self.assertIn("Never cite a passage merely because retrieval returned it", normalized)

    def test_application_prompt_requires_teaching_explanation_and_practical_application(self) -> None:
        transport = FakeTransport(
            api_response(
                answer_content(
                    mode="application",
                    interpretation="The teaching separates effort from outcome.",
                    application="Focus on the next responsible action.",
                )
            )
        )
        provider = GroqLLMProvider(
            client=GroqClient(api_key="offline-test-key", transport=transport)
        )
        answer = provider.generate(
            GenerationRequest(
                self.context,
                AnswerMode.APPLICATION,
                language_policy=self.policy,
            )
        )
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn("APPLICATION MODE QUALITY", prompt)
        self.assertIn("concise scriptural teaching", prompt)
        self.assertIn("Explain clearly what that teaching means", prompt)
        self.assertIn("practical application addressing the user's actual situation", prompt)
        self.assertIn(
            "If evidence_sufficient=true, application must be a non-null, non-empty string",
            prompt,
        )
        self.assertIn("must be derived from the supplied teaching", prompt)
        self.assertIn(
            '"application": "required non-empty application when evidence_sufficient is true"',
            prompt,
        )
        self.assertIn("avoid repetitive verse-by-verse paraphrases", prompt.lower())
        self.assertEqual(answer.mode, AnswerMode.APPLICATION)

    def test_philosophical_prompt_requires_interpretation_when_sufficient(self) -> None:
        transport = FakeTransport(
            api_response(
                answer_content(
                    mode="philosophical",
                    interpretation="A supported philosophical explanation.",
                )
            )
        )
        provider = GroqLLMProvider(
            client=GroqClient(api_key="offline-test-key", transport=transport)
        )
        answer = provider.generate(
            GenerationRequest(
                self.context,
                AnswerMode.PHILOSOPHICAL,
                language_policy=self.policy,
            )
        )
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertIn("PHILOSOPHICAL MODE REQUIREMENTS", prompt)
        self.assertIn(
            "If evidence_sufficient=true, interpretation must be a non-null",
            prompt,
        )
        self.assertIn(
            '"interpretation": "required non-empty interpretation when evidence_sufficient is true"',
            prompt,
        )
        self.assertIn("Set application=null", prompt)
        self.assertEqual(answer.mode, AnswerMode.PHILOSOPHICAL)

    def test_insufficient_evidence_output_remains_supported(self) -> None:
        transport = FakeTransport(
            api_response(
                answer_content(
                    cited_ids=[],
                    evidence_sufficient=False,
                    limitations=[
                        "The supplied evidence addresses duty but not the full situation."
                    ],
                )
            )
        )
        provider = GroqLLMProvider(
            client=GroqClient(api_key="offline-test-key", transport=transport)
        )
        answer = provider.generate(self.request)
        prompt = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertFalse(answer.evidence_sufficient)
        self.assertIn("addresses duty", answer.limitations[0])
        self.assertIn("only partially addresses the question", prompt)
        normalized = " ".join(prompt.split())
        self.assertIn("include a concise non-empty limitation", normalized)
        self.assertIn("answer only the portion directly supported", normalized)
        self.assertIn("Do not fill gaps from model memory", normalized)
        self.assertIn("shorter incomplete-but-grounded answer", normalized)

    def test_urllib_request_includes_all_required_headers(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return FakeHTTPResponse(api_response("{}"))

        with patch(
            "vedavault_retrieval.groq.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            content = GroqClient(api_key="offline-test-key").complete(
                ({"role": "user", "content": "offline request"},),
                max_completion_tokens=1,
            )

        self.assertEqual(content, "{}")
        self.assertEqual(captured["timeout"], 30.0)
        headers = {
            name.lower(): value
            for name, value in captured["request"].header_items()
        }
        self.assertEqual(headers["authorization"], "Bearer offline-test-key")
        self.assertEqual(headers["content-type"], "application/json")
        self.assertEqual(headers["accept"], "application/json")
        self.assertEqual(headers["user-agent"], "VedaVault/1.0 Python")

    def test_urllib_http_403_preserves_safe_groq_diagnostics(self) -> None:
        api_key = "offline-secret-key"
        error_body = json.dumps(
            {
                "error": {
                    "message": f"Access denied for {api_key}",
                    "type": "authentication_error",
                    "code": "access_denied",
                }
            }
        ).encode("utf-8")
        http_error = urllib.error.HTTPError(
            DEFAULT_GROQ_ENDPOINT,
            403,
            "Forbidden",
            hdrs=None,
            fp=io.BytesIO(error_body),
        )
        provider = GroqLLMProvider(client=GroqClient(api_key=api_key))

        with patch(
            "vedavault_retrieval.groq.urllib.request.urlopen",
            side_effect=http_error,
        ), self.assertRaises(LLMProviderError) as raised:
            provider.generate(self.request)

        diagnostic = str(raised.exception)
        self.assertIn("Groq HTTP 403", diagnostic)
        self.assertIn("code=access_denied", diagnostic)
        self.assertIn("type=authentication_error", diagnostic)
        self.assertIn("message=Access denied for [REDACTED]", diagnostic)
        self.assertNotIn(api_key, diagnostic)
        self.assertNotIn(f"Bearer {api_key}", diagnostic)
        self.assertNotIn("Authorization", diagnostic)

    def test_successful_query_rewrite_preserves_query_and_language_policy(self) -> None:
        model_output = json.dumps(
            {
                "retrieval_query": (
                    "Bhagavad Gita teaching on effort, duty, and non-attachment to results"
                ),
                "clarification_required": False,
            }
        )
        transport = FakeTransport(api_response(model_output))
        client = GroqClient(api_key="offline-test-key", transport=transport)
        provider = GroqQueryUnderstandingProvider(client=client)
        original = "gita me agar pura try karu fir bhi result na mile toh kya karu"
        policy = LanguagePolicy(
            input_languages=(SupportedLanguage.HINDI, SupportedLanguage.ENGLISH),
            code_switched=True,
            transliterated=True,
        )
        result = provider.understand(original, policy)
        self.assertEqual(result.original_query, original)
        self.assertIs(result.language_policy, policy)
        self.assertTrue(result.retrieval_query.startswith("Bhagavad Gita teaching"))
        payload = transport.calls[0]["payload"]
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertEqual(payload["max_completion_tokens"], 256)
        system_prompt = payload["messages"][0]["content"]
        self.assertIn("Do not answer the scripture question", system_prompt)
        self.assertIn("Do not provide verse numbers", system_prompt)
        self.assertIn("Do not invent or reproduce quotations", system_prompt)
        self.assertIn("English retrieval query is preferred", system_prompt)
        self.assertIn("not scripture or evidence", system_prompt)
        self.assertIn("exactly these two fields and no others", system_prompt)
        self.assertIn("retrieval_query (a non-empty string)", system_prompt)
        self.assertIn("clarification_required (a boolean)", system_prompt)
        self.assertIn("Do not return original_query", system_prompt)

    def test_language_only_follow_up_resolves_to_minimal_prior_topic(self) -> None:
        model_output = json.dumps(
            {
                "retrieval_query": "Bhagavad Gita teachings on desire",
                "clarification_required": False,
            }
        )
        transport = FakeTransport(api_response(model_output))
        provider = GroqQueryUnderstandingProvider(
            client=GroqClient(api_key="offline-test-key", transport=transport)
        )
        context = ConversationContext(
            (
                ConversationTurn(
                    ConversationRole.USER,
                    "What does the Gita say about desire?",
                ),
                ConversationTurn(
                    ConversationRole.ASSISTANT,
                    "A compact prior explanation.",
                    SupportedLanguage.ENGLISH,
                ),
            )
        )

        result = provider.understand("Explain that in Bengali.", self.policy, context)

        self.assertEqual(result.retrieval_query, "desire")
        call = transport.calls[0]["payload"]
        system_prompt = call["messages"][0]["content"]
        normalized_prompt = " ".join(system_prompt.split())
        user_input = json.loads(call["messages"][1]["content"])
        self.assertEqual(user_input["original_query"], "Explain that in Bengali.")
        self.assertEqual(
            [turn["role"] for turn in user_input["conversation_context"]["recent_turns"]],
            ["user", "assistant"],
        )
        self.assertIn("use it only to resolve references", system_prompt)
        self.assertIn("CURRENT turn", system_prompt)
        self.assertIn("never scripture evidence", system_prompt)
        self.assertIn("shortest faithful standalone", normalized_prompt)
        self.assertIn('"explain that"', normalized_prompt)
        self.assertIn('"in Bengali"', normalized_prompt)
        self.assertIn('"সহজ করে বলো"', normalized_prompt)
        self.assertIn('"Bhagavad Gita teachings on"', normalized_prompt)
        self.assertIn('"Gita says about"', normalized_prompt)
        self.assertIn('"explanation of"', normalized_prompt)

    def test_bengali_language_only_follow_up_resolves_to_prior_topic(self) -> None:
        transport = FakeTransport(
            api_response(
                json.dumps(
                    {
                        "retrieval_query": "desire",
                        "clarification_required": False,
                    }
                )
            )
        )
        provider = GroqQueryUnderstandingProvider(
            client=GroqClient(api_key="offline-test-key", transport=transport)
        )
        context = ConversationContext(
            (
                ConversationTurn(
                    ConversationRole.USER,
                    "What does the Gita say about desire?",
                ),
                ConversationTurn(
                    ConversationRole.ASSISTANT,
                    "A compact prior explanation.",
                    SupportedLanguage.ENGLISH,
                ),
            )
        )
        policy = LanguagePolicy(
            input_languages=(SupportedLanguage.BENGALI,),
            requested_response_language=SupportedLanguage.BENGALI,
        )

        result = provider.understand(
            "এবার বাংলায় বুঝিয়ে বলো",
            policy,
            context,
        )

        self.assertEqual(result.retrieval_query, "desire")
        self.assertIs(result.language_policy, policy)
        user_input = json.loads(
            transport.calls[0]["payload"]["messages"][1]["content"]
        )
        self.assertEqual(
            user_input["language_policy"]["requested_response_language"],
            "bn",
        )

    def test_explicit_current_topic_replaces_old_context_topic(self) -> None:
        transport = FakeTransport(
            api_response(
                json.dumps(
                    {"retrieval_query": "desire", "clarification_required": False}
                )
            )
        )
        provider = GroqQueryUnderstandingProvider(
            client=GroqClient(api_key="offline-test-key", transport=transport)
        )
        context = ConversationContext(
            (
                ConversationTurn(ConversationRole.USER, "What causes anger?"),
                ConversationTurn(
                    ConversationRole.ASSISTANT,
                    "A compact prior explanation.",
                    SupportedLanguage.ENGLISH,
                ),
            )
        )

        result = provider.understand("What about desire?", self.policy, context)

        self.assertEqual(result.retrieval_query, "desire")
        self.assertNotIn("anger", result.retrieval_query.casefold())

    def test_dependent_follow_up_keeps_resolved_topic_and_relation(self) -> None:
        transport = FakeTransport(
            api_response(
                json.dumps(
                    {
                        "retrieval_query": "causes of anger",
                        "clarification_required": False,
                    }
                )
            )
        )
        provider = GroqQueryUnderstandingProvider(
            client=GroqClient(api_key="offline-test-key", transport=transport)
        )
        context = ConversationContext(
            (
                ConversationTurn(ConversationRole.USER, "What is anger?"),
                ConversationTurn(
                    ConversationRole.ASSISTANT,
                    "A compact prior explanation.",
                    SupportedLanguage.ENGLISH,
                ),
            )
        )

        result = provider.understand("Why does it arise?", self.policy, context)

        self.assertEqual(result.retrieval_query, "causes of anger")

    def test_query_understanding_unexpected_fields_remain_rejected(self) -> None:
        model_output = json.dumps(
            {
                "retrieval_query": "Bhagavad Gita teaching on disciplined action",
                "clarification_required": False,
                "unexpected": "must not be accepted",
            }
        )
        transport = FakeTransport(api_response(model_output))
        provider = GroqQueryUnderstandingProvider(
            client=GroqClient(api_key="offline-test-key", transport=transport)
        )

        with self.assertRaisesRegex(
            QueryUnderstandingProviderError,
            "invalid query-understanding JSON",
        ):
            provider.understand("What does the Gita teach?", self.policy)

    def test_malformed_api_response_is_mapped_to_provider_error(self) -> None:
        transport = FakeTransport(b'{"choices":[]}')
        provider = GroqLLMProvider(
            client=GroqClient(api_key="offline-test-key", transport=transport)
        )
        with self.assertRaisesRegex(LLMProviderError, "malformed API response"):
            provider.generate(self.request)

    def test_http_failure_is_mapped_without_raw_exception_leakage(self) -> None:
        transport = FakeTransport(OSError("private upstream detail"))
        provider = GroqLLMProvider(
            client=GroqClient(api_key="offline-test-key", transport=transport)
        )
        with self.assertRaisesRegex(LLMProviderError, "^Groq request failed$") as raised:
            provider.generate(self.request)
        self.assertNotIn("private upstream detail", str(raised.exception))

    def test_malformed_json_model_output_is_rejected_for_both_providers(self) -> None:
        generation = GroqLLMProvider(
            client=GroqClient(
                api_key="offline-test-key",
                transport=FakeTransport(api_response("not-json")),
            )
        )
        rewriting = GroqQueryUnderstandingProvider(
            client=GroqClient(
                api_key="offline-test-key",
                transport=FakeTransport(api_response("[]")),
            )
        )
        with self.assertRaisesRegex(LLMProviderError, "invalid answer JSON"):
            generation.generate(self.request)
        with self.assertRaisesRegex(
            QueryUnderstandingProviderError,
            "invalid query-understanding JSON",
        ):
            rewriting.understand("What is duty?", self.policy)

    def test_answer_citation_mismatch_is_rejected_by_answer_contract(self) -> None:
        transport = FakeTransport(
            api_response(answer_content(cited_ids=["BG_99_99"]))
        )
        provider = GroqLLMProvider(
            client=GroqClient(api_key="offline-test-key", transport=transport)
        )
        with self.assertRaisesRegex(LLMProviderError, "invalid answer JSON"):
            provider.generate(self.request)

    def test_response_evidence_mismatch_is_rejected_by_llm_provider_validation(self) -> None:
        transport = FakeTransport(
            api_response(answer_content(cited_ids=[], evidence_ids=[]))
        )
        provider = GroqLLMProvider(
            client=GroqClient(api_key="offline-test-key", transport=transport)
        )
        with self.assertRaisesRegex(
            LLMProviderError,
            "evidence IDs do not match supplied grounding evidence",
        ):
            provider.generate(self.request)


if __name__ == "__main__":
    unittest.main()
