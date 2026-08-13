"""Normalize the supplied Bhagavad Gita sources into a provenance-preserving corpus."""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RAW_PARENT = ROOT / "Data" / "Raw"
DEFAULT_OUTPUT_PATH = ROOT / "Data" / "Processed" / "Bhagavad_Gita" / "corpus.json"
WORK = "Bhagavad Gita"

SOURCE_JSON = "dharmicdata_json"
SOURCE_BHAGWAD_CSV = "bhagwad_gita_csv"
SOURCE_GEETA_CSV = "geeta_dataset_csv"

SENTINEL_PATTERN = re.compile(
    r"\b(?:no\s+(?:commentary|translation)|did not comment|no comment)\b", re.IGNORECASE
)
CONTAMINATED_FIELDS = {
    (SOURCE_GEETA_CSV, 4, 12, "hindi"),
    (SOURCE_GEETA_CSV, 6, 36, "hindi"),
    (SOURCE_BHAGWAD_CSV, 12, 3, "HinMeaning"),
    (SOURCE_BHAGWAD_CSV, 12, 18, "HinMeaning"),
}

JSON_TRANSLATOR_LANGUAGES = {
    "sri harikrishnadas goenka": "Hindi",
    "swami ramsukhdas": "Hindi",
    "swami tejomayananda": "Hindi",
    "swami adidevananda": "English",
    "swami gambirananda": "English",
    "swami sivananda": "English",
    "dr. s. sankaranarayan": "English",
    "shri purohit swami": "English",
}


def find_raw_directory() -> Path:
    """Find DharmicData without depending on filesystem case preservation."""
    for candidate in RAW_PARENT.iterdir():
        if candidate.is_dir() and candidate.name.casefold() == "dharmicdata":
            return candidate
    raise FileNotFoundError(f"DharmicData directory not found beneath {RAW_PARENT}")


def canonical_id(chapter: int, verse: int) -> str:
    return f"BG_{chapter:02d}_{verse:02d}"


def is_sentinel(value: str) -> bool:
    return bool(SENTINEL_PATTERN.search(value))


def _provenance(source_id: str, raw_file: str, raw_field: str, raw_record_id: str) -> dict[str, str]:
    return {
        "source_id": source_id,
        "raw_file": raw_file,
        "raw_field": raw_field,
        "raw_record_id": raw_record_id,
    }


def _passage(chapter: int, verse: int) -> dict[str, Any]:
    return {
        "passage_id": canonical_id(chapter, verse),
        "work": WORK,
        "chapter": chapter,
        "verse": verse,
        "sanskrit": [],
        "transliteration": [],
        "translations": [],
        "commentaries": [],
        "provenance": [],
        "flags": [],
    }


def _add_text(passage: dict[str, Any], field: str, text: str, provenance: dict[str, str], **metadata: str) -> None:
    passage[field].append({"text": text, **metadata, "provenance": provenance})


def _add_flag(passage: dict[str, Any], code: str, raw_value: str, provenance: dict[str, str]) -> None:
    passage["flags"].append({"code": code, "raw_value": raw_value, "provenance": provenance})


def _record_provenance(passage: dict[str, Any], source_id: str, raw_file: str, raw_record_id: str) -> None:
    passage["provenance"].append(
        {"source_id": source_id, "raw_file": raw_file, "raw_record_id": raw_record_id}
    )


def _add_translation_or_flag(
    passage: dict[str, Any], text: str, language: str, translator: str, provenance: dict[str, str]
) -> None:
    if is_sentinel(text):
        _add_flag(passage, "semantic_missing_value", text, provenance)
        return
    _add_text(passage, "translations", text, provenance, language=language, translator=translator)


def _add_commentary_or_flag(passage: dict[str, Any], text: str, author: str, provenance: dict[str, str]) -> None:
    if is_sentinel(text):
        _add_flag(passage, "semantic_missing_value", text, provenance)
        return
    _add_text(passage, "commentaries", text, provenance, author=author)


def ingest_json(raw_directory: Path, passages: dict[tuple[int, int], dict[str, Any]]) -> set[tuple[int, int]]:
    identifiers: set[tuple[int, int]] = set()
    for path in sorted(raw_directory.glob("bhagavad_gita_chapter_*.json"), key=lambda item: int(re.search(r"(\d+)$", item.stem).group(1))):
        rows = json.loads(path.read_text(encoding="utf-8-sig"))["BhagavadGitaChapter"]
        for row in rows:
            chapter, verse = int(row["chapter"]), int(row["verse"])
            key = (chapter, verse)
            identifiers.add(key)
            passage = passages.setdefault(key, _passage(chapter, verse))
            record_id = f"{chapter}:{verse}"
            _record_provenance(passage, SOURCE_JSON, path.name, record_id)
            _add_text(passage, "sanskrit", row["text"], _provenance(SOURCE_JSON, path.name, "text", record_id))
            for translator, text in row["translations"].items():
                _add_translation_or_flag(
                    passage,
                    text,
                    JSON_TRANSLATOR_LANGUAGES.get(translator, "Unknown"),
                    translator,
                    _provenance(SOURCE_JSON, path.name, f"translations.{translator}", record_id),
                )
            for author, text in row["commentaries"].items():
                _add_commentary_or_flag(
                    passage, text, author, _provenance(SOURCE_JSON, path.name, f"commentaries.{author}", record_id)
                )
    return identifiers


def _read_csv(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def ingest_bhagwad_csv(raw_directory: Path, passages: dict[tuple[int, int], dict[str, Any]]) -> set[tuple[int, int]]:
    path = raw_directory / "Bhagwad_Gita.csv"
    identifiers: set[tuple[int, int]] = set()
    for row in _read_csv(path):
        chapter, verse = int(row["Chapter"]), int(row["Verse"])
        key = (chapter, verse)
        identifiers.add(key)
        passage = passages.setdefault(key, _passage(chapter, verse))
        record_id = row["ID"]
        _record_provenance(passage, SOURCE_BHAGWAD_CSV, path.name, record_id)
        _add_text(passage, "sanskrit", row["Shloka"], _provenance(SOURCE_BHAGWAD_CSV, path.name, "Shloka", record_id))
        _add_text(passage, "transliteration", row["Transliteration"], _provenance(SOURCE_BHAGWAD_CSV, path.name, "Transliteration", record_id))
        for field, language, translator in (
            ("HinMeaning", "Hindi", "Bhagwad_Gita.csv"),
            ("EngMeaning", "English", "Bhagwad_Gita.csv"),
            ("WordMeaning", "English", "Bhagwad_Gita.csv word meaning"),
        ):
            provenance = _provenance(SOURCE_BHAGWAD_CSV, path.name, field, record_id)
            _add_translation_or_flag(passage, row[field], language, translator, provenance)
            if (SOURCE_BHAGWAD_CSV, chapter, verse, field) in CONTAMINATED_FIELDS:
                _add_flag(passage, "contaminated_language_field", row[field], provenance)
    return identifiers


def ingest_geeta_csv(raw_directory: Path, passages: dict[tuple[int, int], dict[str, Any]]) -> set[tuple[int, int]]:
    path = raw_directory / "geeta_dataset.csv"
    identifiers: set[tuple[int, int]] = set()
    for row in _read_csv(path):
        chapter, verse = int(row["chapter"]), int(row["verse"])
        key = (chapter, verse)
        identifiers.add(key)
        passage = passages.setdefault(key, _passage(chapter, verse))
        record_id = f"{chapter}:{verse}"
        _record_provenance(passage, SOURCE_GEETA_CSV, path.name, record_id)
        _add_text(passage, "sanskrit", row["sanskrit"], _provenance(SOURCE_GEETA_CSV, path.name, "sanskrit", record_id))
        _add_text(passage, "transliteration", row["transliteration"], _provenance(SOURCE_GEETA_CSV, path.name, "transliteration", record_id))
        for field, language in (("hindi", "Hindi"), ("english", "English")):
            provenance = _provenance(SOURCE_GEETA_CSV, path.name, field, record_id)
            _add_translation_or_flag(passage, row[field], language, "geeta_dataset.csv", provenance)
            if (SOURCE_GEETA_CSV, chapter, verse, field) in CONTAMINATED_FIELDS:
                _add_flag(passage, "contaminated_language_field", row[field], provenance)
    return identifiers


def build_corpus(raw_directory: Path | None = None) -> dict[str, Any]:
    """Read all sources and return the deterministic normalized corpus."""
    raw_directory = raw_directory or find_raw_directory()
    passages: dict[tuple[int, int], dict[str, Any]] = {}
    source_identifiers = {
        SOURCE_JSON: ingest_json(raw_directory, passages),
        SOURCE_BHAGWAD_CSV: ingest_bhagwad_csv(raw_directory, passages),
        SOURCE_GEETA_CSV: ingest_geeta_csv(raw_directory, passages),
    }
    expected = next(iter(source_identifiers.values()))
    if any(identifiers != expected for identifiers in source_identifiers.values()):
        raise ValueError("source verse identifiers are not aligned")
    return {
        "schema_version": 1,
        "work": WORK,
        "source_alignment": {
            "key": ["chapter", "verse"],
            "source_record_counts": {source_id: len(ids) for source_id, ids in source_identifiers.items()},
        },
        "passages": [passages[key] for key in sorted(passages)],
    }


def write_corpus(output_path: Path = DEFAULT_OUTPUT_PATH) -> Path:
    corpus = build_corpus()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(corpus, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT_PATH
    written_path = write_corpus(output_path)
    print(f"Wrote normalized corpus: {written_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
