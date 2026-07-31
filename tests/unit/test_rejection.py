"""Unit tests for ``orchestration/rejection.py``'s ``build_rejection_reason``.

A pure function — no orchestrator, no fixtures — deriving a one-line
explanation of why a draft missed the quality gate from its own Trust/ATS
reports and the gate's thresholds.
"""

from __future__ import annotations

from trustresume.models import ATSReport, QualityGate, TrustReport
from trustresume.orchestration import build_rejection_reason


def test_buildRejectionReason_bothScoresFail_mentionsBoth() -> None:
    gate = QualityGate()  # min_trust=90, min_ats=85
    trust = TrustReport(claims=[], score=62.0)
    ats = ATSReport(score=78.0)

    reason = build_rejection_reason(gate, trust, ats)

    assert "Trust score 62" in reason
    assert "needs >= 90" in reason
    assert "ATS score 78" in reason
    assert "needs >= 85" in reason


def test_buildRejectionReason_onlyAtsFails_mentionsOnlyAts() -> None:
    gate = QualityGate()
    trust = TrustReport(claims=[], score=95.0)
    ats = ATSReport(score=78.0)

    reason = build_rejection_reason(gate, trust, ats)

    assert "Trust score" not in reason
    assert "ATS score 78" in reason


def test_buildRejectionReason_onlyTrustFails_mentionsOnlyTrust() -> None:
    gate = QualityGate()
    trust = TrustReport(claims=[], score=62.0)
    ats = ATSReport(score=90.0)

    reason = build_rejection_reason(gate, trust, ats)

    assert "Trust score 62" in reason
    assert "ATS score" not in reason


def test_buildRejectionReason_neitherScoreFails_fallsBackToGenericMessage() -> None:
    """Not reachable from the real _persist call site (only invoked when
    ``not state.passed``), but must still return something sane rather than
    an empty string if ever called on a technically-passing pair.
    """
    gate = QualityGate()
    trust = TrustReport(claims=[], score=95.0)
    ats = ATSReport(score=90.0)

    reason = build_rejection_reason(gate, trust, ats)

    assert reason
    assert "Trust score" not in reason
    assert "ATS score" not in reason


def test_buildRejectionReason_respectsCustomGateThresholds() -> None:
    gate = QualityGate(min_trust_score=50, min_ats_score=50)
    trust = TrustReport(claims=[], score=40.0)
    ats = ATSReport(score=90.0)

    reason = build_rejection_reason(gate, trust, ats)

    assert "needs >= 50" in reason
    assert "ATS score" not in reason
