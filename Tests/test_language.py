"""Offline tests for multilingual conversation policy contracts, not language detection."""

from __future__ import annotations

import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Backend"))

from vedavault_retrieval import (  # noqa: E402
    AnswerMode,
    EvidenceBundle,
    GenerationRequest,
    GroundingContext,
    LanguagePolicy,
    RetrievalDocument,
    RetrievalResult,
    SUPPORTED_LANGUAGES,
    SupportedLanguage,
    WritingScript,
)


class LanguagePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        evidence = EvidenceBundle.from_retrieval(
            "Explain this teaching.",
            [
                RetrievalResult(
                    RetrievalDocument(
                        "first",
                        "Karmanye vadhikaraste.",
                        {
                            "passage_id": "BG_02_47",
                            "chapter": 2,
                            "verse": 47,
                            "text_layer": "sanskrit",
                            "source": {"source_id": "gita-source"},
                        },
                    ),
                    0.91,
                )
            ],
        )
        self.context = GroundingContext.from_evidence_bundle(evidence)

    def test_supported_languages_are_stable_and_duplicate_input_languages_are_rejected(self) -> None:
        self.assertEqual(
            SUPPORTED_LANGUAGES,
            (
                SupportedLanguage.ENGLISH,
                SupportedLanguage.HINDI,
                SupportedLanguage.BENGALI,
                SupportedLanguage.SANSKRIT,
                SupportedLanguage.TAMIL,
                SupportedLanguage.TELUGU,
                SupportedLanguage.MARATHI,
                SupportedLanguage.GUJARATI,
            ),
        )
        for language in SUPPORTED_LANGUAGES:
            policy = LanguagePolicy(input_languages=(language,))
            self.assertEqual(policy.effective_primary_response_language, language)
        with self.assertRaises(ValueError):
            LanguagePolicy(input_languages=(SupportedLanguage.HINDI, SupportedLanguage.HINDI))

    def test_response_language_precedence_is_explicit_then_current_then_conversation_then_english(self) -> None:
        explicit = LanguagePolicy(
            input_languages=(SupportedLanguage.BENGALI,),
            conversation_language=SupportedLanguage.HINDI,
            requested_response_language=SupportedLanguage.ENGLISH,
        )
        current = LanguagePolicy(
            input_languages=(SupportedLanguage.BENGALI,),
            conversation_language=SupportedLanguage.HINDI,
        )
        conversation = LanguagePolicy(conversation_language=SupportedLanguage.HINDI)
        fallback = LanguagePolicy()
        self.assertEqual(explicit.effective_primary_response_language, SupportedLanguage.ENGLISH)
        self.assertEqual(current.effective_primary_response_language, SupportedLanguage.BENGALI)
        self.assertEqual(conversation.effective_primary_response_language, SupportedLanguage.HINDI)
        self.assertEqual(fallback.effective_primary_response_language, SupportedLanguage.ENGLISH)
        with self.assertRaises(TypeError):
            LanguagePolicy(default_response_language=SupportedLanguage.HINDI)  # type: ignore[call-arg]

    def test_secondary_response_language_never_changes_primary_and_rejects_all_primary_collisions(self) -> None:
        policy = LanguagePolicy(
            input_languages=(SupportedLanguage.BENGALI,),
            secondary_response_language=SupportedLanguage.ENGLISH,
        )
        self.assertEqual(policy.effective_primary_response_language, SupportedLanguage.BENGALI)
        self.assertEqual(policy.secondary_response_language, SupportedLanguage.ENGLISH)
        collisions = (
            {
                "requested_response_language": SupportedLanguage.ENGLISH,
                "secondary_response_language": SupportedLanguage.ENGLISH,
            },
            {
                "input_languages": (SupportedLanguage.BENGALI,),
                "secondary_response_language": SupportedLanguage.BENGALI,
            },
            {
                "conversation_language": SupportedLanguage.HINDI,
                "secondary_response_language": SupportedLanguage.HINDI,
            },
            {"secondary_response_language": SupportedLanguage.ENGLISH},
        )
        for values in collisions:
            with self.subTest(values=values), self.assertRaises(ValueError):
                LanguagePolicy(**values)

    def test_public_policy_field_types_are_validated(self) -> None:
        invalid_policies = (
            {"input_languages": "bn"},
            {"input_languages": ("bn",)},
            {"conversation_language": "hi"},
            {"requested_response_language": "en"},
            {"secondary_response_language": "en"},
            {"script_hint": "Latn"},
            {"code_switched": 1},
            {"transliterated": "false"},
            {"clarification_needed": None},
            {"clarification_reason": 1},
        )
        for values in invalid_policies:
            with self.subTest(values=values), self.assertRaises(ValueError):
                LanguagePolicy(**values)

    def test_script_language_transliteration_and_code_switching_are_independent(self) -> None:
        romanized_hindi = LanguagePolicy(
            input_languages=(SupportedLanguage.HINDI, SupportedLanguage.ENGLISH),
            code_switched=True,
            transliterated=True,
            script_hint=WritingScript.LATIN,
        )
        romanized_sanskrit = LanguagePolicy(
            input_languages=(SupportedLanguage.SANSKRIT,),
            transliterated=True,
            script_hint=WritingScript.LATIN,
        )
        devanagari_sanskrit = LanguagePolicy(
            input_languages=(SupportedLanguage.SANSKRIT,),
            script_hint=WritingScript.DEVANAGARI,
        )
        self.assertEqual(romanized_hindi.current_input_language, SupportedLanguage.HINDI)
        self.assertNotEqual(romanized_hindi.current_input_language, SupportedLanguage.ENGLISH)
        self.assertTrue(romanized_hindi.code_switched)
        self.assertTrue(romanized_hindi.transliterated)
        self.assertEqual(romanized_sanskrit.script_hint, WritingScript.LATIN)
        self.assertEqual(devanagari_sanskrit.script_hint, WritingScript.DEVANAGARI)
        self.assertEqual(devanagari_sanskrit.current_input_language, SupportedLanguage.SANSKRIT)

    def test_input_languages_are_defensively_immutable(self) -> None:
        mutable_languages = [SupportedLanguage.HINDI, SupportedLanguage.ENGLISH]
        policy = LanguagePolicy(input_languages=mutable_languages)  # type: ignore[arg-type]
        mutable_languages.append(SupportedLanguage.BENGALI)
        self.assertEqual(policy.input_languages, (SupportedLanguage.HINDI, SupportedLanguage.ENGLISH))
        self.assertIsInstance(policy.input_languages, tuple)
        with self.assertRaises(FrozenInstanceError):
            policy.input_languages = ()  # type: ignore[misc]

    def test_clarification_is_explicit_boolean_metadata(self) -> None:
        normal = LanguagePolicy(input_languages=(SupportedLanguage.ENGLISH,))
        ambiguous = LanguagePolicy(
            input_languages=(SupportedLanguage.ENGLISH,),
            clarification_needed=True,
            clarification_reason="The referent is not available in the current context.",
        )
        self.assertFalse(normal.clarification_needed)
        self.assertTrue(ambiguous.clarification_needed)
        with self.assertRaises(ValueError):
            LanguagePolicy(clarification_needed=True)
        with self.assertRaises(ValueError):
            LanguagePolicy(clarification_reason="A reason without a clarification state")

    def test_serialization_is_complete_and_detached_from_the_policy(self) -> None:
        policy = LanguagePolicy(
            input_languages=[SupportedLanguage.HINDI, SupportedLanguage.ENGLISH],  # type: ignore[arg-type]
            conversation_language=SupportedLanguage.BENGALI,
            requested_response_language=SupportedLanguage.SANSKRIT,
            secondary_response_language=SupportedLanguage.ENGLISH,
            code_switched=True,
            transliterated=True,
            script_hint=WritingScript.LATIN,
            clarification_needed=True,
            clarification_reason="The requested tradition is unclear.",
        )
        expected = {
            "input_languages": ["hi", "en"],
            "conversation_language": "bn",
            "requested_response_language": "sa",
            "effective_primary_response_language": "sa",
            "secondary_response_language": "en",
            "code_switched": True,
            "transliterated": True,
            "script_hint": "Latn",
            "clarification_needed": True,
            "clarification_reason": "The requested tradition is unclear.",
        }
        serialized = policy.to_dict()
        self.assertEqual(serialized, expected)
        serialized["input_languages"].append("ta")  # type: ignore[index]
        serialized["conversation_language"] = "changed"
        self.assertEqual(policy.input_languages, (SupportedLanguage.HINDI, SupportedLanguage.ENGLISH))
        self.assertEqual(policy.conversation_language, SupportedLanguage.BENGALI)
        self.assertEqual(policy.to_dict(), expected)
        self.assertEqual(
            LanguagePolicy().to_dict(),
            {
                "input_languages": [],
                "conversation_language": None,
                "requested_response_language": None,
                "effective_primary_response_language": "en",
                "secondary_response_language": None,
                "code_switched": False,
                "transliterated": False,
                "script_hint": None,
                "clarification_needed": False,
                "clarification_reason": None,
            },
        )

    def test_generation_request_remains_backward_compatible_and_query_is_grounded(self) -> None:
        request_without_policy = GenerationRequest(self.context, AnswerMode.TEXTUAL)
        policy = LanguagePolicy(input_languages=(SupportedLanguage.BENGALI,))
        request_with_policy = GenerationRequest(self.context, AnswerMode.TEXTUAL, language_policy=policy)
        self.assertIsNone(request_without_policy.language_policy)
        self.assertEqual(request_without_policy.query, self.context.query)
        self.assertEqual(request_with_policy.query, self.context.query)
        self.assertIs(request_with_policy.language_policy, policy)
        self.assertFalse(hasattr(policy, "raw_query"))
        self.assertEqual(request_with_policy.grounding_context.evidence_items[0].text_layer, "sanskrit")


if __name__ == "__main__":
    unittest.main()
