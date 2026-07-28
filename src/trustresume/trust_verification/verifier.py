"""Reusable Trust Harness logic: prompt, formatting, and report assembly.

Extracted from the Trust Harness agent so the verification workflow lives in
one place independent of the LLM wrapper (ADR-0004). The agent supplies the
model and calls these; anything else that needs to format a draft/evidence for
verification or assemble a scored report reuses the same functions.

Layering note: the 0–100 scoring *rubric* stays on ``TrustReport`` (it's
data-level — classified claims in, number out). This module owns everything
*around* it: how a draft and its evidence are presented to the model, and how
the classified claims become a report.

Milestone M6 (trust verification + evaluation).
"""

from __future__ import annotations

from trustresume.models import EvidenceSet, ResumeDraft, TrustReport, VerifiedClaim

# The fact-checker instruction. Kept here (not in the agent) so the prompt is
# versioned alongside the rest of the verification logic.
SYSTEM_PROMPT = """\
You are a resume fact-checker. You are given a resume draft and the candidate \
evidence it was supposed to be based on. Extract every discrete factual claim \
the draft makes (skills, experience, certifications, achievements). For each \
claim, decide whether the evidence SUPPORTS it, PARTIALLY_SUPPORTS it, or does \
not support it (UNSUPPORTED). Cite the evidence you relied on. Be strict: if \
the evidence does not clearly back a claim, it is not SUPPORTED. Do not give \
the draft the benefit of the doubt."""


def format_draft(draft: ResumeDraft) -> str:
    """Render a draft as plain text for the fact-checker prompt."""
    lines = [draft.summary] if draft.summary else []
    for section in draft.sections:
        lines.append(f"# {section.heading}")
        lines.extend(f"- {b}" for b in section.bullets)
    return "\n".join(lines) or "(empty draft)"


def format_evidence(evidence: EvidenceSet) -> str:
    """Render evidence chunks (with ids, for citation) for the prompt."""
    if not evidence.chunks:
        return "(no candidate evidence)"
    return "\n".join(f"- [{c.chunk_id}] {c.text}" for c in evidence.chunks)


def build_prompt(draft: ResumeDraft, evidence: EvidenceSet) -> str:
    """Compose the full verification prompt from a draft and its evidence."""
    return (
        f"## Resume draft\n{format_draft(draft)}\n\n"
        f"## Candidate evidence\n{format_evidence(evidence)}"
    )


def build_trust_report(claims: list[VerifiedClaim], *, iteration: int) -> TrustReport:
    """Assemble a scored :class:`TrustReport` from classified claims.

    Applies the model's deterministic rubric — the score is never taken from
    the LLM (ADR-0004).
    """
    return TrustReport.from_claims(claims, iteration=iteration)
