"""Model-independent contract and validation for future grounded answers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

from .grounding import GroundingContext


ANSWER_CONTRACT_RULES = """ANSWER CONTRACT RULES
- Scriptural teaching must cite supplied canonical verse identifiers.
- Scriptural teaching, interpretation, and application are distinct fields.
- Interpretation and application must not be presented as direct scripture.
- Quotations and verse references must not be fabricated.
- Insufficient evidence must be acknowledged through limitations."""


class AnswerMode(str, Enum):
    """The intended scope of a future answer, not a generation strategy."""

    TEXTUAL = "textual"
    PHILOSOPHICAL = "philosophical"
    APPLICATION = "application"


@dataclass(frozen=True)
class ScripturalClaim:
    """A future answer's supported statement of scriptural teaching, never a quotation."""

    statement: str
    cited_verse_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.statement, str) or not self.statement.strip():
            raise ValueError("scriptural claim statement must be a non-empty string")
        citations = tuple(self.cited_verse_ids)
        if not citations or not all(isinstance(citation, str) and citation.strip() for citation in citations):
            raise ValueError("scriptural claims require non-empty canonical verse citations")
        object.__setattr__(self, "cited_verse_ids", citations)


@dataclass(frozen=True)
class AnswerContract:
    """Validated future-answer structure; this class does not generate content."""

    query: str
    mode: AnswerMode
    scriptural_claims: tuple[ScripturalClaim, ...] = ()
    interpretation: str | None = None
    application: str | None = None
    evidence_sufficient: bool = True
    limitations: tuple[str, ...] = ()
    evidence_passage_ids: frozenset[str] = field(default_factory=frozenset, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(self.mode, AnswerMode):
            raise ValueError("mode must be an AnswerMode")
        claims = tuple(self.scriptural_claims)
        limitations = tuple(self.limitations)
        evidence_ids = frozenset(self.evidence_passage_ids)
        if not all(isinstance(claim, ScripturalClaim) for claim in claims):
            raise ValueError("scriptural_claims must contain ScripturalClaim values")
        if not all(isinstance(value, str) and value.strip() for value in limitations):
            raise ValueError("limitations must contain non-empty strings")
        if not all(isinstance(passage_id, str) and passage_id.strip() for passage_id in evidence_ids):
            raise ValueError("evidence passage IDs must be non-empty strings")
        if not self.evidence_sufficient and not limitations:
            raise ValueError("insufficient evidence requires at least one limitation")
        if any(citation not in evidence_ids for claim in claims for citation in claim.cited_verse_ids):
            raise ValueError("scriptural claim cites a verse absent from supplied evidence")
        if self.mode is AnswerMode.TEXTUAL and (self.interpretation is not None or self.application is not None):
            raise ValueError("textual mode cannot include interpretation or application")
        if self.mode is AnswerMode.PHILOSOPHICAL:
            if self.application is not None:
                raise ValueError("philosophical mode cannot include application")
            if self.evidence_sufficient and not _non_empty(self.interpretation):
                raise ValueError("philosophical mode requires interpretation when evidence is sufficient")
        if self.mode is AnswerMode.APPLICATION and self.evidence_sufficient and not _non_empty(self.application):
            raise ValueError("application mode requires application when evidence is sufficient")
        object.__setattr__(self, "scriptural_claims", claims)
        object.__setattr__(self, "limitations", limitations)
        object.__setattr__(self, "evidence_passage_ids", evidence_ids)

    @classmethod
    def from_grounding_context(
        cls,
        context: GroundingContext,
        mode: AnswerMode,
        scriptural_claims: tuple[ScripturalClaim, ...] = (),
        interpretation: str | None = None,
        application: str | None = None,
        evidence_sufficient: bool = True,
        limitations: tuple[str, ...] = (),
    ) -> "AnswerContract":
        """Build a validated contract using only passage IDs present in grounding evidence."""
        evidence_passage_ids = frozenset(
            item.passage_id for item in context.evidence_items if item.passage_id is not None
        )
        return cls(
            query=context.query,
            mode=mode,
            scriptural_claims=scriptural_claims,
            interpretation=interpretation,
            application=application,
            evidence_sufficient=evidence_sufficient,
            limitations=limitations,
            evidence_passage_ids=evidence_passage_ids,
        )

    @property
    def cited_verse_ids(self) -> tuple[str, ...]:
        """All claim citations in deterministic first-use order."""
        return tuple(dict.fromkeys(citation for claim in self.scriptural_claims for citation in claim.cited_verse_ids))

    def to_json(self) -> str:
        """Serialize the contract deterministically for a future provider-neutral answer layer."""
        return json.dumps(
            {
                "query": self.query,
                "mode": self.mode.value,
                "scriptural_teaching": [
                    {"statement": claim.statement, "cited_verse_ids": list(claim.cited_verse_ids)}
                    for claim in self.scriptural_claims
                ],
                "interpretation": self.interpretation,
                "application": self.application,
                "cited_verse_ids": list(self.cited_verse_ids),
                "evidence_sufficient": self.evidence_sufficient,
                "limitations": list(self.limitations),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def _non_empty(value: str | None) -> bool:
    return isinstance(value, str) and bool(value.strip())
