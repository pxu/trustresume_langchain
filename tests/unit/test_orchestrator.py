"""Unit tests for the Orchestrator and rewrite-feedback generation.

The orchestrator's job is *control flow*, so it's tested against lightweight
fake agents (same ``run`` shape as the real ones) whose scripted outputs let us
drive the quality loop deterministically: pass on first try, fail-then-pass,
and fail-to-cap. This isolates the loop logic from LLM behavior — exactly the
visibility ADR-0003 is after.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

from trustresume.models import (
    ATSReport,
    CandidateProfile,
    ClaimCategory,
    ClaimStatus,
    EvidenceSet,
    JobDescription,
    QualityGate,
    ResumeDraft,
    TrustReport,
    VerifiedClaim,
)
from trustresume.orchestration import Orchestrator, build_feedback

_T = TypeVar("_T")


def run(awaitable: Awaitable[_T]) -> _T:
    return asyncio.run(awaitable)  # type: ignore[arg-type]


# --- fakes matching each agent's run() shape -------------------------------


class FakeJobDescriptionAgent:
    async def run(self, job_posting: str) -> JobDescription:
        return JobDescription(raw_text=job_posting, title="Engineer", keywords=["python"])


class FakeCandidateProfileService:
    """Records how many times it was asked to resolve a profile."""

    def __init__(self) -> None:
        self.calls = 0

    async def get_or_refresh(self, user_id: str) -> CandidateProfile:
        self.calls += 1
        return CandidateProfile(name="Jordan Rivera", skills=["python"])


class FakeRetrievalAgent:
    async def run(self, *, user_id: str, job: JobDescription) -> EvidenceSet:
        return EvidenceSet(user_id=user_id, query="python", chunks=[])


class FakeResumeAgent:
    """Records each call so we can assert how many rewrites happened."""

    def __init__(self) -> None:
        self.calls: list[str | None] = []

    async def run(self, *, job, evidence, feedback=None, iteration=0):  # type: ignore[no-untyped-def]
        self.calls.append(feedback)
        return ResumeDraft(summary=f"draft {iteration}", iteration=iteration)


class ScriptedTrustAgent:
    """Returns a pre-scripted Trust score per iteration (last value repeats)."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    async def run(self, *, draft: ResumeDraft, evidence: EvidenceSet) -> TrustReport:
        score = self._scores[min(draft.iteration, len(self._scores) - 1)]
        status = ClaimStatus.SUPPORTED if score >= 90 else ClaimStatus.UNSUPPORTED
        claim = VerifiedClaim(text="Knows Python", category=ClaimCategory.SKILL, status=status)
        return TrustReport(claims=[claim], score=score, iteration=draft.iteration)


class ScriptedATSAgent:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    async def run(self, *, draft: ResumeDraft, job: JobDescription) -> ATSReport:
        score = self._scores[min(draft.iteration, len(self._scores) - 1)]
        missing = [] if score >= 85 else ["kubernetes"]
        return ATSReport(score=score, missing_keywords=missing, iteration=draft.iteration)


def _make(
    trust_scores: list[float], ats_scores: list[float]
) -> tuple[Orchestrator, FakeResumeAgent]:
    resume = FakeResumeAgent()
    orch = Orchestrator(
        job_description_agent=FakeJobDescriptionAgent(),  # type: ignore[arg-type]
        candidate_profile_service=FakeCandidateProfileService(),  # type: ignore[arg-type]
        retrieval_agent=FakeRetrievalAgent(),  # type: ignore[arg-type]
        resume_agent=resume,  # type: ignore[arg-type]
        trust_agent=ScriptedTrustAgent(trust_scores),  # type: ignore[arg-type]
        evaluation_agent=ScriptedATSAgent(ats_scores),  # type: ignore[arg-type]
    )
    return orch, resume


# --- orchestrator control flow ---------------------------------------------


def test_orchestrator_passesOnFirstTry_noRewrites() -> None:
    orch, resume = _make(trust_scores=[95.0], ats_scores=[90.0])

    state = run(orch.run(user_id="u1", job_posting="Python role"))

    assert state.passed is True
    assert state.iteration == 0
    assert len(state.drafts) == 1
    assert resume.calls == [None]  # writer called once, no feedback


def test_orchestrator_failsThenPasses_oneRewrite() -> None:
    # iteration 0 fails on trust, iteration 1 passes both.
    orch, resume = _make(trust_scores=[80.0, 95.0], ats_scores=[90.0, 90.0])

    state = run(orch.run(user_id="u1", job_posting="Python role"))

    assert state.passed is True
    assert state.iteration == 1
    assert len(state.drafts) == 2
    # Second writer call received feedback derived from the failing reports.
    assert resume.calls[0] is None
    assert resume.calls[1] is not None


def test_orchestrator_failsToCap_stopsAndExportsRealScores() -> None:
    orch, resume = _make(trust_scores=[10.0], ats_scores=[10.0])  # always fails

    state = run(orch.run(user_id="u1", job_posting="Python role", gate=QualityGate()))

    assert state.passed is False
    assert state.is_exhausted is True
    # Initial + 3 rewrites = 4 drafts (cap is 3 iterations).
    assert state.iteration == 3
    assert len(state.drafts) == 4
    # Real, un-fudged scores survive on the capped-out draft.
    assert state.current_trust.score == 10.0  # type: ignore[union-attr]
    assert state.current_ats.score == 10.0  # type: ignore[union-attr]


def test_orchestrator_resolvesCandidateProfileOnce_evenAcrossRewrites() -> None:
    orch, _resume = _make(trust_scores=[80.0, 95.0], ats_scores=[90.0, 90.0])

    state = run(orch.run(user_id="u1", job_posting="Python role"))

    assert state.candidate_profile is not None
    assert state.candidate_profile.name == "Jordan Rivera"
    assert orch._candidate_profile.calls == 1  # type: ignore[attr-defined]


def test_orchestrator_respectsCustomGate() -> None:
    # A lenient gate passes what the default would reject.
    orch, _resume = _make(trust_scores=[70.0], ats_scores=[70.0])

    state = run(
        orch.run(
            user_id="u1",
            job_posting="role",
            gate=QualityGate(min_trust_score=60, min_ats_score=60),
        )
    )
    assert state.passed is True
    assert state.iteration == 0


# --- feedback generation ----------------------------------------------------


def test_buildFeedback_listsHallucinationsAndMissingKeywords() -> None:
    trust = TrustReport(
        claims=[
            VerifiedClaim(
                text="Knows Kubernetes",
                category=ClaimCategory.SKILL,
                status=ClaimStatus.UNSUPPORTED,
            ),
            VerifiedClaim(text="Knows Python", status=ClaimStatus.SUPPORTED),
        ],
        score=50.0,
    )
    ats = ATSReport(score=70.0, missing_keywords=["aws"])

    feedback = build_feedback(trust, ats)
    assert "Kubernetes" in feedback
    assert "aws" in feedback
    assert "Python" not in feedback  # supported claims aren't flagged


def test_buildFeedback_scoreOnlyFailure_stillActionable() -> None:
    trust = TrustReport(claims=[], score=40.0)
    ats = ATSReport(score=80.0)
    feedback = build_feedback(trust, ats)
    assert "Trust 40" in feedback
