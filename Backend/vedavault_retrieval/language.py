"""Model-independent multilingual conversation policy; no detection or translation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SupportedLanguage(str, Enum):
    """VedaVault V1 language policy codes (ISO 639-1 where assigned)."""

    ENGLISH = "en"
    HINDI = "hi"
    BENGALI = "bn"
    SANSKRIT = "sa"
    TAMIL = "ta"
    TELUGU = "te"
    MARATHI = "mr"
    GUJARATI = "gu"


class WritingScript(str, Enum):
    """Optional Unicode/script hints; these are evidence, never language identification."""

    LATIN = "Latn"
    DEVANAGARI = "Deva"
    BENGALI = "Beng"
    TAMIL = "Taml"
    TELUGU = "Telu"
    GUJARATI = "Gujr"
    MIXED = "mixed"


SUPPORTED_LANGUAGES = tuple(SupportedLanguage)


@dataclass(frozen=True)
class LanguagePolicy:
    """Immutable declared language/conversation metadata for one user turn.

    This value object does not detect language, normalize text, translate, or
    alter evidence. `input_languages` is an ordered set of declared or
    future-detector-provided hints: the first is the clearly established
    current-turn language used for response resolution. The raw query belongs
    to the retrieval, grounding, and generation-request flow.
    """

    input_languages: tuple[SupportedLanguage, ...] = ()
    conversation_language: SupportedLanguage | None = None
    requested_response_language: SupportedLanguage | None = None
    secondary_response_language: SupportedLanguage | None = None
    code_switched: bool = False
    transliterated: bool = False
    script_hint: WritingScript | None = None
    clarification_needed: bool = False
    clarification_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.input_languages, tuple | list):
            raise ValueError("input_languages must be an ordered collection of SupportedLanguage values")
        input_languages = tuple(self.input_languages)
        if not all(isinstance(language, SupportedLanguage) for language in input_languages):
            raise ValueError("input_languages must contain SupportedLanguage values")
        if len(set(input_languages)) != len(input_languages):
            raise ValueError("input_languages must not contain duplicates")
        for field_name in (
            "conversation_language",
            "requested_response_language",
            "secondary_response_language",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, SupportedLanguage):
                raise ValueError(f"{field_name} must be a SupportedLanguage or None")
        for field_name in ("code_switched", "transliterated", "clarification_needed"):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be boolean")
        if self.script_hint is not None and not isinstance(self.script_hint, WritingScript):
            raise ValueError("script_hint must be a WritingScript or None")
        if self.clarification_reason is not None and not isinstance(self.clarification_reason, str):
            raise ValueError("clarification_reason must be a string or None")
        if self.clarification_needed and not _non_empty(self.clarification_reason):
            raise ValueError("clarification_needed requires a non-empty clarification_reason")
        if not self.clarification_needed and self.clarification_reason is not None:
            raise ValueError("clarification_reason requires clarification_needed")
        object.__setattr__(self, "input_languages", input_languages)
        if self.secondary_response_language == self.effective_primary_response_language:
            raise ValueError("secondary_response_language must differ from the primary response language")

    @property
    def current_input_language(self) -> SupportedLanguage | None:
        """The first current-turn language hint, if one is established."""
        return self.input_languages[0] if self.input_languages else None

    @property
    def effective_primary_response_language(self) -> SupportedLanguage:
        """Resolve response language: explicit override, current turn, conversation, then English."""
        return (
            self.requested_response_language
            or self.current_input_language
            or self.conversation_language
            or SupportedLanguage.ENGLISH
        )

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic, provider-neutral policy representation."""
        return {
            "input_languages": [language.value for language in self.input_languages],
            "conversation_language": _code(self.conversation_language),
            "requested_response_language": _code(self.requested_response_language),
            "effective_primary_response_language": self.effective_primary_response_language.value,
            "secondary_response_language": _code(self.secondary_response_language),
            "code_switched": self.code_switched,
            "transliterated": self.transliterated,
            "script_hint": self.script_hint.value if self.script_hint is not None else None,
            "clarification_needed": self.clarification_needed,
            "clarification_reason": self.clarification_reason,
        }


def _code(language: SupportedLanguage | None) -> str | None:
    return language.value if language is not None else None


def _non_empty(value: str | None) -> bool:
    return isinstance(value, str) and bool(value.strip())
