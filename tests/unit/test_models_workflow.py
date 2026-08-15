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


def test_workflowState_passingButUnderCap_stillContinues() -> None:
    """No early exit on a pass — the loop always runs to the iteration cap."""
    state = _state_with_scores(trust=92.0, ats=88.0, iteration=1)  # default cap is 3

    assert state.passed is True
    assert state.should_continue is True


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


def _append_draft(state: WorkflowState, *, trust: float, ats: float, iteration: int) -> None:
    state.drafts.append(ResumeDraft(iteration=iteration))
    state.trust_reports.append(TrustReport(claims=[], score=trust, iteration=iteration))
    state.ats_reports.append(ATSReport(score=ats, iteration=iteration))
    state.iteration = iteration


def test_workflowState_freshRun_hasNoFinalDraft() -> None:
    """final_* mirror current_* on an unscored run: nothing to ship yet."""
    state = WorkflowState(user_id="u1")

    assert state.final_index is None
    assert state.final_draft is None
    assert state.final_trust is None
    assert state.final_ats is None
    assert state.final_passed is False


def test_workflowState_singlePassingDraft_finalIsThatDraft() -> None:
    state = _state_with_scores(trust=92.0, ats=88.0, iteration=0)

    assert state.final_index == 0
    assert state.final_passed is True


def test_workflowState_passingDraftBeatsHigherAtsFailingDraft() -> None:
    """A pass must never be demoted just because a failing draft has a higher ATS score.

    This is a real, reachable scenario now that the loop always keeps
    generating drafts after an early pass: a later, failing rewrite with
    strong keyword coverage but a trust regression must not outrank an
    earlier draft that genuinely passed.
    """
    state = _state_with_scores(trust=50.0, ats=99.0, iteration=0)  # fails on trust, huge ATS
    _append_draft(state, trust=91.0, ats=86.0, iteration=1)  # passes both, lower ATS

    assert state.final_index == 1
    assert state.final_passed is True


def test_workflowState_allFailed_shipsHighestAtsNotLast() -> None:
    """The whole point: a rewrite that scored lower doesn't overwrite the best draft."""
    state = _state_with_scores(trust=88.0, ats=84.0, iteration=0)  # close, but fails
    _append_draft(state, trust=40.0, ats=50.0, iteration=1)  # regressed rewrite

    assert state.current_draft is state.drafts[1]  # latest is the regressed one
    assert state.final_index == 0  # ...but we ship the earlier, higher-ATS draft
    assert state.final_ats is not None and state.final_ats.score == 84.0
    assert state.final_passed is False  # still a fail — just the best of the failures


def test_workflowState_allFailed_ranksByAtsOnly_ignoringTrust() -> None:
    """Among drafts that all failed the gate, only ATS breaks the tie — not Trust.

    A deliberate product choice, not an oversight: there is no Trust-based
    fallback once every draft has failed, so a draft with far worse Trust but
    better keyword coverage is still selected here.
    """
    state = _state_with_scores(trust=89.0, ats=60.0, iteration=0)  # nearly passes on trust
    _append_draft(state, trust=20.0, ats=99.0, iteration=1)  # terrible trust, huge ATS coverage

    assert state.final_index == 1


def test_workflowState_allFailed_tieOnAts_goesToLaterIterationRegardlessOfTrust() -> None:
    """Equal ATS resolves to the more-refined rewrite — Trust plays no role in the tie."""
    state = _state_with_scores(trust=99.0, ats=70.0, iteration=0)  # higher trust, same ATS
    _append_draft(state, trust=10.0, ats=70.0, iteration=1)  # far lower trust, same ATS

    assert state.final_index == 1  # later iteration wins despite far lower trust
