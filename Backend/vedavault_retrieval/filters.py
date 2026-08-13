"""Metadata filter matching for retrieval results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


MetadataFilter = Mapping[str, Any]


def matches_metadata(metadata: Mapping[str, Any], filters: MetadataFilter | None) -> bool:
    """Match exact scalar values or membership in a supplied list/set/tuple."""
    if not filters:
        return True
    for key, expected in filters.items():
        actual = metadata.get(key)
        if isinstance(expected, (list, set, tuple, frozenset)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True
