"""Trust Harness agent — draft + evidence -> ``TrustReport`` (RQ3, ADR-0004).

Fourth step and the project's primary research contribution. Runs as its own
LLM pass *after* generation: it extracts each discrete claim from the draft and
classifies it against the retrieved evidence as SUPPORTED / PARTIALLY_SUPPORTED
/ UNSUPPORTED, citing the evidence chunks it relied on.

Two deliberate separations:
- The Resume Writer is never trusted to self-report accuracy; verification is a
  distinct step with its own model pass (ADR-0004).
- The 0-100 **Trust Score is computed by deterministic code**, not emitted by
  the LLM — the model's job is to classify claims; the rubric is ours.

This agent is a thin LLM wrapper: the prompt, formatting, and report assembly
live in ``trustresume.trust_verification`` (M6) so they're reusable and testable
without a model.

Milestone M4 (agents); verification logic extracted in M6.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from trustresume.models import EvidenceSet, ResumeDraft, TrustReport, VerifiedClaim
from trustresume.trust_verification import (
    SYSTEM_PROMPT,
    build_prompt,
    build_trust_report,
)

from .base import ModelInput, ensure_type, with_structured_retry


class _ClaimExtraction(BaseModel):
    """LLM output: the classified claims (the score is computed in code)."""

    claims: list[VerifiedClaim] = Field(default_factory=list)


class TrustHarnessAgent:
    """Verifies a draft's claims against evidence and scores its trustworthiness."""

    def __init__(self, model: ModelInput) -> None:
        self._structured = with_structured_retry(model, _ClaimExtraction)

    async def run(
        self,
        *,
        draft: ResumeDraft,
        evidence: EvidenceSet,
    ) -> TrustReport:
        """Classify the draft's claims and compute its Trust Score."""
        result = await self._structured.ainvoke(
            [SystemMessage(SYSTEM_PROMPT), HumanMessage(build_prompt(draft, evidence))]
        )
        claims = ensure_type(result, _ClaimExtraction).claims
        claims = _drop_unknown_citations(claims, evidence)
        return build_trust_report(claims, iteration=draft.iteration)


def _drop_unknown_citations(
    claims: list[VerifiedClaim], evidence: EvidenceSet
) -> list[VerifiedClaim]:
    """Strip any cited chunk id the model invented rather than actually retrieved.

    The whole point of ``evidence_chunk_ids`` is an auditable trail from a
    verdict back to real evidence (ADR-0004) — an unchecked id (hallucinated,
    or copied from a different run) would be indistinguishable from a real
    citation to anything reading the report. This doesn't downgrade the
    claim's status (the model may have genuinely relied on evidence it cited
    correctly alongside one bad id), it only removes citations that don't
    resolve to a chunk actually in ``evidence``.
    """
    known_ids = {c.chunk_id for c in evidence.chunks}
    return [
        claim.model_copy(
            update={
                "evidence_chunk_ids": [cid for cid in claim.evidence_chunk_ids if cid in known_ids]
            }
        )
        for claim in claims
    ]
