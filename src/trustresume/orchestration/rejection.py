"""Rejection-reason generation for a draft that failed the quality gate.

Separate from ``feedback.py`` deliberately: ``build_feedback`` writes
rewrite instructions for the *next LLM pass* (an internal, mid-loop
audience); this module writes a short, human-readable explanation of *why*
the final, capped-out draft failed (an external, persisted-and-displayed
audience) — different purpose, different reader, kept as its own function
rather than overloading ``build_feedback``'s contract.

Milestone M5 (orchestration); added post-M5 alongside resume persistence.
"""

from __future__ import annotations

from trustresume.models import ATSReport, QualityGate, TrustReport


def build_rejection_reason(gate: QualityGate, trust: TrustReport, ats: ATSReport) -> str:
    """A one-line explanation of why a draft didn't pass ``gate``.

    Reports whichever threshold(s) the final draft actually missed — only
    Trust, only ATS, or both — so a caller doesn't have to re-derive which
    score(s) failed from the raw numbers.
    """
    misses: list[str] = []
    if trust.score < gate.min_trust_score:
        misses.append(f"Trust score {trust.score:.0f} (needs >= {gate.min_trust_score:.0f})")
    if ats.score < gate.min_ats_score:
        misses.append(f"ATS score {ats.score:.0f} (needs >= {gate.min_ats_score:.0f})")
    if not misses:
        # Both scores technically clear the gate but the draft still didn't
        # "pass" by the time the caller asked for a rejection reason — only
        # reachable if this is called on a state that wasn't actually
        # rejected; state that plainly rather than implying a false score gap.
        return (
            "Draft did not pass the quality gate for reasons other than the trust/ATS thresholds."
        )
    return "; ".join(misses) + "."
