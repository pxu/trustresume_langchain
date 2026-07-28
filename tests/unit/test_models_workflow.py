"""Unit tests for the orchestrator WorkflowState and QualityGate."""

from __future__ import annotations

from trustresume.models import (
    ATSReport,
    QualityGate,
    ResumeDraft,
    TrustReport,
    WorkflowState,
)


def test_qualityGate_defaults_matchAdr0005() -> None:
    gate = QualityGate()
    assert gate.min_trust_score == 90.0
    assert gate.min_ats_score == 85.0
    assert gate.max_iterations == 3


def test_qualityGate_passes_requiresBothScores() -> None:
    gate = QualityGate()
    high_trust = TrustReport(claims=[], score=95.0)
    low_trust = TrustReport(claims=[], score=80.0)
    high_ats = ATSReport(score=90.0)
    low_ats = ATSReport(score=70.0)

    assert gate.passes(high_trust, high_ats) is True
    assert gate.passes(low_trust, high_ats) is False
    assert gate.passes(high_trust, low_ats) is False


def _state_with_scores(trust: float, ats: float, iteration: int) -> WorkflowState:
    state = WorkflowState(user_id="u1")
    state.drafts.append(ResumeDraft(iteration=iteration))
    state.trust_reports.append(TrustReport(claims=[], score=trust, iteration=iteration))
    state.ats_reports.append(ATSReport(score=ats, iteration=iteration))
    state.iteration = iteration
    return state


def test_workflowState_freshRun_shouldContinue() -> None:
    state = WorkflowState(user_id="u1")

    assert state.current_draft is None
    assert state.passed is False
    assert state.should_continue is True


def test_workflowState_passingDraft_stops() -> None:
    state = _state_with_scores(trust=92.0, ats=88.0, iteration=1)

    assert state.passed is True
    assert state.should_continue is False


def test_workflowState_failingUnderCap_continues() -> None:
    state = _state_with_scores(trust=80.0, ats=88.0, iteration=1)

    assert state.passed is False
    assert state.is_exhausted is False
    assert state.should_continue is True


def test_workflowState_failingAtCap_stopsAndExports() -> None:
    state = _state_with_scores(trust=80.0, ats=88.0, iteration=3)

    assert state.passed is False
    assert state.is_exhausted is True
    # ADR-0005: a capped-out failing draft still stops the loop (and is exported).
    assert state.should_continue is False
    assert state.current_draft is not None


def test_workflowState_currentAccessors_returnLatest() -> None:
    state = _state_with_scores(trust=80.0, ats=70.0, iteration=1)
    state.drafts.append(ResumeDraft(iteration=2))
    state.trust_reports.append(TrustReport(claims=[], score=91.0, iteration=2))
    state.ats_reports.append(ATSReport(score=86.0, iteration=2))
    state.iteration = 2

    assert state.current_draft is not None and state.current_draft.iteration == 2
    assert state.current_trust is not None and state.current_trust.score == 91.0
    assert state.current_ats is not None and state.current_ats.score == 86.0
    assert state.passed is True
