"""Tests for the source manifest validator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Scripts"))
from validate_sources import DEFAULT_MANIFEST_PATH, validate_manifest


class ValidateSourcesTests(unittest.TestCase):
    def _write_manifest(self, content: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "sources.yaml"
        path.write_text(content, encoding="utf-8")
        return path

    def test_canonical_manifest_is_valid(self) -> None:
        self.assertEqual(validate_manifest(DEFAULT_MANIFEST_PATH), [])

    def test_missing_required_field_is_reported(self) -> None:
        path = self._write_manifest(
            "sources:\n"
            "  - id: incomplete-source\n"
            "    title: Incomplete Source\n"
            "    tradition: Example\n"
            "    language: English\n"
            "    source_url: https://example.test/source\n"
            "    license: CC0-1.0\n"
            "    content_path: Data/Raw/example.txt\n"
            "    format: text\n"
        )
        self.assertIn("incomplete-source: missing required field 'status'", validate_manifest(path))

    def test_duplicate_ids_are_reported(self) -> None:
        entry = (
            "  - id: repeated\n"
            "    title: Example\n"
            "    tradition: Example\n"
            "    language: English\n"
            "    source_url: https://example.test/source\n"
            "    license: CC0-1.0\n"
            "    content_path: Data/Raw/example.txt\n"
            "    format: text\n"
            "    status: approved\n"
        )
        path = self._write_manifest("sources:\n" + entry + entry)
        self.assertIn("duplicate source id: repeated", validate_manifest(path))


if __name__ == "__main__":
    unittest.main()
