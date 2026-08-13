"""Validate VedaVault's constrained YAML source manifest without dependencies."""

from __future__ import annotations

import re
import sys
from pathlib import Path


REQUIRED_FIELDS = (
    "id",
    "title",
    "tradition",
    "language",
    "source_url",
    "license",
    "content_path",
    "format",
    "status",
)

FIELD_PATTERN = re.compile(r"^    ([a-z_]+):(?:[ ](.*))?$")
DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "Data" / "sources.yaml"


def _scalar(value: str) -> str:
    """Return a simple YAML scalar, allowing matching single or double quotes."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def parse_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Parse the limited YAML layout deliberately used by Data/sources.yaml."""
    errors: list[str] = []
    sources: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    in_sources = False

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line == "sources:":
            in_sources = True
            continue
        if not in_sources:
            continue
        if line.startswith("  - "):
            current = {}
            sources.append(current)
            first_field = line[4:]
            if first_field:
                key, separator, value = first_field.partition(":")
                if not separator or not key:
                    errors.append(f"line {line_number}: invalid source field")
                else:
                    current[key] = _scalar(value)
            continue
        match = FIELD_PATTERN.match(line)
        if match and current is not None:
            current[match.group(1)] = _scalar(match.group(2) or "")
        else:
            errors.append(f"line {line_number}: unsupported manifest syntax")

    if not in_sources:
        errors.append("missing top-level 'sources:' list")
    if not sources:
        errors.append("manifest contains no source entries")
    return sources, errors


def validate_manifest(path: Path) -> list[str]:
    """Return validation errors; an empty list means the manifest is valid."""
    if not path.is_file():
        return [f"manifest not found: {path}"]

    sources, errors = parse_manifest(path)
    seen_ids: set[str] = set()
    for index, source in enumerate(sources, 1):
        label = source.get("id", f"entry {index}")
        for field in REQUIRED_FIELDS:
            if not source.get(field, "").strip():
                errors.append(f"{label}: missing required field '{field}'")
        source_id = source.get("id", "").strip()
        if source_id and source_id in seen_ids:
            errors.append(f"duplicate source id: {source_id}")
        seen_ids.add(source_id)
    return errors


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MANIFEST_PATH
    errors = validate_manifest(path)
    if errors:
        print("Manifest validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"Manifest valid: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
