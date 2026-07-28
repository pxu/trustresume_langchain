"""Trust Harness output: per-claim verification and a 0-100 Trust Score.

This is the schema behind the project's primary research contribution (RQ3,
ADR-0004). The Trust Harness extracts discrete claims from a draft, retrieves
evidence for each, classifies it SUPPORTED / PARTIALLY_SUPPORTED /
UNSUPPORTED, flags hallucinations, and rolls the results up into a Trust Score.

The score is carried as data the harness computes and sets, not derived
implicitly on read — the harness owns the scoring rubric (which may evolve),
and the model just needs to hold and validate the result. A ``compute_score``
helper is provided for a simple, transparent default rubric.

Milestone M1 (shared models).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import ClaimCategory, ClaimStatus


class VerifiedClaim(BaseModel):
    """One claim extracted from the draft, plus its evidence-based verdict.

    ``evidence_chunk_ids`` references the ``EvidenceChunk.chunk_id`` values the
    harness used, keeping the verdict explainable and auditable.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, description="The claim as stated in the resume draft.")
    category: ClaimCategory = Field(
        default=ClaimCategory.OTHER,
        description="Kind of assertion the claim makes.",
    )
    status: ClaimStatus = Field(..., description="Evidence-based classification of the claim.")
    evidence_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Ids of the evidence chunks supporting this verdict.",
    )
    rationale: str | None = Field(
        None,
        description="Short explanation of why the claim got this status.",
    )

    @property
    def is_hallucination(self) -> bool:
        """True when an assertion of fact is entirely unsupported by evidence.

        Unsupported skills, experience, certifications, and achievements are
        flagged as hallucinations (ADR-0004); an UNSUPPORTED ``OTHER`` claim
        (e.g. a stylistic phrase) is not.
        """
        return self.status is ClaimStatus.UNSUPPORTED and self.category is not ClaimCategory.OTHER


class TrustReport(BaseModel):
    """Verification result for one resume draft.

    ``score`` (0-100) is the headline Trust Score; ``iteration`` matches the
    draft it was computed for so the orchestrator can pair reports with drafts
    across the quality loop.
    """

    model_config = ConfigDict(extra="forbid")

    claims: list[VerifiedClaim] = Field(
        default_factory=list,
        description="Every claim extracted from the draft, with its verdict.",
    )
    score: float = Field(..., ge=0, le=100, description="0-100 Trust Score for the draft.")
    iteration: int = Field(
        default=0,
        ge=0,
        description="Quality-loop iteration this report corresponds to.",
    )

    @property
    def hallucinations(self) -> list[VerifiedClaim]:
        """Claims flagged as hallucinations (see ``VerifiedClaim.is_hallucination``)."""
        return [c for c in self.claims if c.is_hallucination]

    @property
    def supported_fraction(self) -> float:
        """Fraction of claims classified SUPPORTED (0.0 when there are no claims)."""
        if not self.claims:
            return 0.0
        supported = sum(1 for c in self.claims if c.status is ClaimStatus.SUPPORTED)
        return supported / len(self.claims)

    @staticmethod
    def compute_score(claims: list[VerifiedClaim]) -> float:
        """A simple, transparent default rubric for the Trust Score.

        SUPPORTED counts full, PARTIALLY_SUPPORTED half, UNSUPPORTED zero;
        averaged and scaled to 0-100. An empty claim set scores 0 - a draft
        with nothing verifiable is not trustworthy. The harness may override
        this with a more sophisticated rubric; it exists so the model can
        stand alone in tests and early wiring.
        """
        if not claims:
            return 0.0
        weights = {
            ClaimStatus.SUPPORTED: 1.0,
            ClaimStatus.PARTIALLY_SUPPORTED: 0.5,
            ClaimStatus.UNSUPPORTED: 0.0,
        }
        total = sum(weights[c.status] for c in claims)
        return round(100.0 * total / len(claims), 2)

    @classmethod
    def from_claims(cls, claims: list[VerifiedClaim], iteration: int = 0) -> TrustReport:
        """Build a report, scoring the claims with the default rubric."""
        return cls(claims=claims, score=cls.compute_score(claims), iteration=iteration)


class ATSReport(BaseModel):
    """ATS Evaluation output: a 0-100 ATS Score and its supporting detail.

    Produced by the ATS Evaluation agent, which scores keyword coverage and
    format compatibility of a draft against the target job description.
    """

    model_config = ConfigDict(extra="forbid")

    score: float = Field(..., ge=0, le=100, description="0-100 ATS Score for the draft.")
    matched_keywords: list[str] = Field(
        default_factory=list,
        description="Job keywords found in the draft.",
    )
    missing_keywords: list[str] = Field(
        default_factory=list,
        description="Job keywords absent from the draft.",
    )
    notes: str | None = Field(None, description="Free-text formatting/compatibility notes.")
    iteration: int = Field(
        default=0,
        ge=0,
        description="Quality-loop iteration this report corresponds to.",
    )

    @model_validator(mode="after")
    def _keywords_disjoint(self) -> ATSReport:
        """A keyword cannot be both matched and missing."""
        overlap = set(self.matched_keywords) & set(self.missing_keywords)
        if overlap:
            raise ValueError(f"keywords appear in both matched and missing: {sorted(overlap)}")
        return self
