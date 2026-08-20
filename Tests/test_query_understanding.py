"""Offline tests for model-independent query understanding."""

from __future__ import annotations

import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Backend"))

from vedavault_retrieval import (  # noqa: E402
    LanguagePolicy,
    QueryUnderstandingProvider,
    QueryUnderstandingResult,
    SupportedLanguage,
    WritingScript,
)


class MappingQueryUnderstandingProvider(QueryUnderstandingProvider):
    """Deterministic test double; no model or network is involved."""

    def __init__(self, rewrites: dict[str, tuple[str, bool]]) -> None:
        self.rewrites = rewrites

    def understand(
        self,
        original_query: str,
        language_policy: LanguagePolicy,
    ) -> QueryUnderstandingResult:
        retrieval_query, clarification_required = self.rewrites[original_query]
        result = QueryUnderstandingResult(
            original_query,
            retrieval_query,
            language_policy,
            clarification_required,
        )
        return self.validate_response(original_query, language_policy, result)


class QueryUnderstandingContractTests(unittest.TestCase):
    def test_result_is_immutable_and_preserves_original_query_exactly(self) -> None:
        policy = LanguagePolicy(input_languages=(SupportedLanguage.HINDI,))
        original = "  gita me kya karu?  "
        result = QueryUnderstandingResult(
            original,
            "Bhagavad Gita teaching on right action",
            policy,
        )
        self.assertEqual(result.original_query, original)
        self.assertIs(result.language_policy, policy)
        with self.assertRaises(FrozenInstanceError):
            result.retrieval_query = "changed"  # type: ignore[misc]

    def test_retrieval_query_policy_and_clarification_are_validated(self) -> None:
        policy = LanguagePolicy(input_languages=(SupportedLanguage.ENGLISH,))
        normal = QueryUnderstandingResult("What is duty?", "Bhagavad Gita teaching on duty", policy)
        ambiguous = QueryUnderstandingResult(
            "What does that mean?",
            "Bhagavad Gita meaning of an unspecified teaching",
            policy,
            clarification_required=True,
        )
        self.assertFalse(normal.clarification_required)
        self.assertTrue(ambiguous.clarification_required)
        self.assertEqual(ambiguous.to_dict()["language_policy"], policy.to_dict())
        with self.assertRaises(ValueError):
            QueryUnderstandingResult("question", " ", policy)
        with self.assertRaises(ValueError):
            QueryUnderstandingResult("question", "retrieval intent", "en")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            QueryUnderstandingResult("question", "retrieval intent", policy, 1)  # type: ignore[arg-type]

    def test_fake_provider_covers_v1_input_shapes_without_answering(self) -> None:
        cases = (
            (
                "How can I act without fearing the result?",
                LanguagePolicy(input_languages=(SupportedLanguage.ENGLISH,)),
                "Bhagavad Gita teaching on action without attachment to results",
            ),
            (
                "फल की चिंता किए बिना कर्म कैसे करूँ?",
                LanguagePolicy(
                    input_languages=(SupportedLanguage.HINDI,),
                    script_hint=WritingScript.DEVANAGARI,
                ),
                "Bhagavad Gita teaching on action without attachment to results",
            ),
            (
                "ফলের চিন্তা না করে কীভাবে কাজ করব?",
                LanguagePolicy(
                    input_languages=(SupportedLanguage.BENGALI,),
                    script_hint=WritingScript.BENGALI,
                ),
                "Bhagavad Gita teaching on action without attachment to results",
            ),
            (
                "कर्मण्येवाधिकारस्ते का तात्पर्य क्या है?",
                LanguagePolicy(
                    input_languages=(SupportedLanguage.SANSKRIT, SupportedLanguage.HINDI),
                    code_switched=True,
                    script_hint=WritingScript.DEVANAGARI,
                ),
                "Meaning of karmany evadhikaras te and non-attachment to results",
            ),
            (
                "gita me agar pura try karu fir bhi result na mile toh kya karu",
                LanguagePolicy(
                    input_languages=(SupportedLanguage.HINDI, SupportedLanguage.ENGLISH),
                    code_switched=True,
                    transliterated=True,
                    script_hint=WritingScript.LATIN,
                ),
                "Bhagavad Gita teaching on effort, duty, and non-attachment to results",
            ),
            (
                "ami try kori but result na pele ki korbo",
                LanguagePolicy(
                    input_languages=(SupportedLanguage.BENGALI, SupportedLanguage.ENGLISH),
                    code_switched=True,
                    transliterated=True,
                    script_hint=WritingScript.LATIN,
                ),
                "Bhagavad Gita teaching on effort and detachment from outcomes",
            ),
            (
                "karmanye vadhikaraste mane ki",
                LanguagePolicy(
                    input_languages=(SupportedLanguage.SANSKRIT,),
                    transliterated=True,
                    script_hint=WritingScript.LATIN,
                ),
                "Meaning of karmany evadhikaras te and duty without attachment",
            ),
            (
                "dharma follow korle bhi fear keno hoy",
                LanguagePolicy(
                    input_languages=(
                        SupportedLanguage.BENGALI,
                        SupportedLanguage.HINDI,
                        SupportedLanguage.ENGLISH,
                    ),
                    code_switched=True,
                    transliterated=True,
                    script_hint=WritingScript.LATIN,
                ),
                "Bhagavad Gita teaching on dharma and overcoming fear",
            ),
            (
                "hw cn i cntrl my mnd",
                LanguagePolicy(input_languages=(SupportedLanguage.ENGLISH,)),
                "Bhagavad Gita teaching on controlling the mind",
            ),
        )
        provider = MappingQueryUnderstandingProvider(
            {original: (retrieval, False) for original, _, retrieval in cases}
        )
        for original, policy, expected_retrieval in cases:
            with self.subTest(original=original):
                result = provider.understand(original, policy)
                self.assertEqual(result.original_query, original)
                self.assertEqual(result.retrieval_query, expected_retrieval)
                self.assertIs(result.language_policy, policy)
                self.assertNotIn("BG_", result.retrieval_query)

    def test_explicit_response_language_override_survives_english_rewrite(self) -> None:
        original = "আমাকে English-এ answer দাও: ফল নিয়ে চিন্তা কেন হয়?"
        policy = LanguagePolicy(
            input_languages=(SupportedLanguage.BENGALI, SupportedLanguage.ENGLISH),
            requested_response_language=SupportedLanguage.ENGLISH,
            code_switched=True,
        )
        provider = MappingQueryUnderstandingProvider(
            {
                original: (
                    "Bhagavad Gita teaching on anxiety and attachment to outcomes",
                    False,
                )
            }
        )
        result = provider.understand(original, policy)
        self.assertIs(result.language_policy, policy)
        self.assertEqual(
            result.language_policy.effective_primary_response_language,
            SupportedLanguage.ENGLISH,
        )


if __name__ == "__main__":
    unittest.main()
