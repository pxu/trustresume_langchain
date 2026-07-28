"""Rewrite-feedback generation for the quality loop.

When a draft fails the gate (ADR-0005), the orchestrator needs *specific*
feedback to hand back to the Resume Writer — "remove the unsupported
Kubernetes claim", "add the AWS keyword" — not just "try again". This is
deterministic (derived from the Trust and ATS reports), so it's cheap,
reproducible, and testable without an LLM.

Milestone M5 (orchestration).
"""

from __future__ import annotations

from trustresume.models import ATSReport, TrustReport


def build_feedback(trust: TrustReport, ats: ATSReport) -> str:
    """Turn failing reports into concrete instructions for the next rewrite.

    Trust issues come first because accuracy outranks keyword coverage — the
    project's whole premise is that an ATS-optimized lie is worse than an
    honest gap.
    """
    lines: list[str] = []

    hallucinations = trust.hallucinations
    if hallucinations:
        lines.append(
            "Remove or rephrase these claims — they are not supported by the candidate's evidence:"
        )
        lines.extend(f"- {c.text} ({c.category.value})" for c in hallucinations)

    partial = [c for c in trust.claims if c.status.value == "PARTIALLY_SUPPORTED"]
    if partial:
        lines.append("Soften these claims to match what the evidence actually supports:")
        lines.extend(f"- {c.text}" for c in partial)

    if ats.missing_keywords:
        lines.append(
            "Where the evidence genuinely supports them, incorporate these "
            "target keywords the draft is missing: " + ", ".join(ats.missing_keywords)
        )

    if not lines:
        # Gate failed on scores alone (e.g. too few supported claims) without a
        # specific actionable item — give the writer the scores to aim past.
        lines.append(
            f"Improve overall evidence-grounding and relevance "
            f"(Trust {trust.score:.0f}, ATS {ats.score:.0f})."
        )

    return "\n".join(lines)
