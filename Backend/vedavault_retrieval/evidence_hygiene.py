"""Deterministic provenance policy applied only when constructing grounding evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .evidence import EvidenceBundle, EvidenceItem


CONTAMINATED_TRANSLATION_PROVENANCE = frozenset(
    {
        ("bhagwad_gita_csv", "WordMeaning"),
        ("dharmicdata_json", "translations.swami adidevananda"),
        ("dharmicdata_json", "translations.swami gambirananda"),
        ("dharmicdata_json", "translations.sri harikrishnadas goenka"),
    }
)

APPROVED_TRANSLATION_PROVENANCE = {
    "English": (
        ("dharmicdata_json", "translations.swami sivananda"),
        ("geeta_dataset_csv", "english"),
        ("dharmicdata_json", "translations.shri purohit swami"),
        ("dharmicdata_json", "translations.dr. s. sankaranarayan"),
        ("bhagwad_gita_csv", "EngMeaning"),
    ),
    "Hindi": (
        ("dharmicdata_json", "translations.swami tejomayananda"),
        ("geeta_dataset_csv", "hindi"),
        ("dharmicdata_json", "translations.swami ramsukhdas"),
        ("bhagwad_gita_csv", "HinMeaning"),
    ),
}

DEFAULT_CORPUS_PATH = (
    Path(__file__).resolve().parents[2]
    / "Data"
    / "Processed"
    / "Bhagavad_Gita"
    / "corpus.json"
)


class EvidenceHygieneError(RuntimeError):
    """Grounding cannot safely use a selected item."""


@dataclass(frozen=True)
class CleanTranslation:
    text: str
    language: str
    translator: str | None
    provenance: Mapping[str, Any]


class EvidenceHygienePolicy:
    """Replace known contaminated selections without changing their rank or verse."""

    def __init__(self, passages: Sequence[Mapping[str, Any]]) -> None:
        self._translations = _approved_translations(passages)

    @classmethod
    def from_corpus_path(cls, corpus_path: Path) -> "EvidenceHygienePolicy":
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        passages = corpus.get("passages")
        if not isinstance(passages, list):
            raise ValueError("canonical corpus must contain a 'passages' list")
        return cls(passages)

    def sanitize(self, bundle: EvidenceBundle) -> tuple[EvidenceItem, ...]:
        """Return grounding-safe items in exactly the selected order."""
        return tuple(self._sanitize_item(item) for item in bundle.items)

    def _sanitize_item(self, item: EvidenceItem) -> EvidenceItem:
        contaminated = _contaminated_provenance(item)
        if contaminated is None:
            return item
        if item.passage_id is None:
            raise EvidenceHygieneError(
                "contaminated evidence has no canonical passage_id; grounding refused"
            )

        language = _replacement_language(item, contaminated)
        replacement = self._translations.get((item.passage_id, language))
        if replacement is None:
            raise EvidenceHygieneError(
                f"no approved {language} translation for contaminated evidence "
                f"{item.passage_id}; grounding refused"
            )

        selection_provenance = {
            "document_id": item.document_id,
            "text_layer": item.text_layer,
            "source": dict(item.source) if item.source is not None else None,
            "provenance": _mapping_copy(item.metadata.get("provenance")),
        }
        grounding_provenance = dict(replacement.provenance)
        metadata = dict(item.metadata)
        metadata.update(
            {
                "language": replacement.language,
                "text_layer": "translations",
                "source": grounding_provenance,
                "provenance": grounding_provenance,
                "selection_provenance": selection_provenance,
                "grounding_provenance": grounding_provenance,
                "evidence_hygiene": {
                    "substituted": True,
                    "reason": "selected translation provenance is not approved for direct grounding",
                },
            }
        )
        if replacement.translator is not None:
            metadata["translator"] = replacement.translator
        else:
            metadata.pop("translator", None)

        return EvidenceItem(
            document_id=item.document_id,
            passage_id=item.passage_id,
            chapter=item.chapter,
            verse=item.verse,
            text=replacement.text,
            text_layer="translations",
            score=item.score,
            source=grounding_provenance,
            metadata=metadata,
        )


@lru_cache(maxsize=1)
def default_evidence_hygiene_policy() -> EvidenceHygienePolicy:
    """Load the checked-in canonical corpus only when grounding first needs it."""
    return EvidenceHygienePolicy.from_corpus_path(DEFAULT_CORPUS_PATH)


def _approved_translations(
    passages: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], CleanTranslation]:
    approved: dict[tuple[str, str], CleanTranslation] = {}
    for passage in passages:
        passage_id = passage.get("passage_id")
        translations = passage.get("translations", ())
        if not isinstance(passage_id, str) or not isinstance(translations, Sequence):
            continue
        candidates: dict[tuple[str, str], CleanTranslation] = {}
        for translation in translations:
            if not isinstance(translation, Mapping):
                continue
            text = translation.get("text")
            language = translation.get("language")
            provenance = translation.get("provenance")
            if (
                not isinstance(text, str)
                or not text.strip()
                or language not in APPROVED_TRANSLATION_PROVENANCE
                or not isinstance(provenance, Mapping)
            ):
                continue
            pair = _provenance_pair(provenance)
            if pair not in APPROVED_TRANSLATION_PROVENANCE[language]:
                continue
            translator = translation.get("translator")
            candidates.setdefault(
                pair,
                CleanTranslation(
                    text=text,
                    language=language,
                    translator=translator if isinstance(translator, str) else None,
                    provenance=dict(provenance),
                ),
            )
        for language, preference in APPROVED_TRANSLATION_PROVENANCE.items():
            for pair in preference:
                if pair in candidates:
                    approved[(passage_id, language)] = candidates[pair]
                    break
    return approved


def _contaminated_provenance(item: EvidenceItem) -> tuple[str, str] | None:
    for value in (item.source, item.metadata.get("provenance")):
        if isinstance(value, Mapping):
            pair = _provenance_pair(value)
            if pair in CONTAMINATED_TRANSLATION_PROVENANCE:
                return pair
    return None


def is_contaminated_evidence(item: EvidenceItem) -> bool:
    """Return whether an item carries any confirmed contaminated provenance."""
    return _contaminated_provenance(item) is not None


def _replacement_language(
    item: EvidenceItem, contaminated: tuple[str, str]
) -> str:
    language = item.metadata.get("language")
    if isinstance(language, str) and language.casefold() == "hindi":
        return "Hindi"
    if contaminated == (
        "dharmicdata_json",
        "translations.sri harikrishnadas goenka",
    ):
        return "Hindi"
    return "English"


def _provenance_pair(provenance: Mapping[str, Any]) -> tuple[str, str]:
    source_id = provenance.get("source_id")
    raw_field = provenance.get("raw_field")
    return (
        source_id if isinstance(source_id, str) else "",
        raw_field if isinstance(raw_field, str) else "",
    )


def _mapping_copy(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None
