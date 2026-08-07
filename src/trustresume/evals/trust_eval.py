"""Trust Harness evaluation: does the verifier actually classify claims right?

This is the project's central correctness claim — "every statement is checked
against the candidate's own evidence" — and until now nothing verified the
verifier. The runtime Trust score measures a *draft*; it is computed from
whatever the harness said, so a harness that rubber-stamps everything produces
a perfect score and an undetectable failure.

Here each labeled case pins one claim against known evidence with a known
correct verdict, so the harness's own accuracy becomes a number that moves
when a prompt or model changes.

Added post-port; no equivalent in the original.
"""

from __future__ import annotations

import logging
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from trustresume.models import (
    ClaimStatus,
    EvidenceChunk,
    EvidenceSet,
    ResumeDraft,
    ResumeSection,
    TrustReport,
)

from .datasets import TrustCase
from .metrics import ClassificationMetrics, classification_metrics
from .retrieval_eval import EVAL_USER_ID

logger = logging.getLogger(__name__)

#: Worst-to-best. ``_dominant_status`` collapses a multi-claim report by
#: taking the *worst* verdict: if the harness split one labeled claim into
#: parts and found any part unsupported, the claim as stated isn't supported.
#: Ordering it this way (rather than "most frequent") keeps the harness honest
#: in the safe direction — it can't pass a claim by burying one bad part
#: among several fine ones.
_SEVERITY = [ClaimStatus.UNSUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED, ClaimStatus.SUPPORTED]


class SupportsTrustRun(Protocol):
    """The Trust Harness surface an evaluation needs (the real agent, or a fake)."""

    async def run(self, *, draft: ResumeDraft, evidence: EvidenceSet) -> TrustReport: ...


class TrustCaseResult(BaseModel):
    """One case's expected vs. actual verdict."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    claim: str
    expected: ClaimStatus
    predicted: ClaimStatus
    note: str | None = None

    @property
    def correct(self) -> bool:
        return self.expected == self.predicted

    @property
    def is_dangerous_error(self) -> bool:
        """True when the harness was *too lenient* — the failure that matters.

        Calling an unsupported claim supported lets a fabrication through to
        the user, which is the exact harm this project exists to prevent.
        The opposite error (flagging a true claim) only costs a rewrite
        iteration, so the two are not equally bad and shouldn't be reported
        as one undifferentiated error rate.
        """
        return _SEVERITY.index(self.predicted) > _SEVERITY.index(self.expected)


class TrustEvalReport(BaseModel):
    """Classification metrics plus per-case detail and a lenience count."""

    model_config = ConfigDict(extra="forbid")

    metrics: ClassificationMetrics
    cases: list[TrustCaseResult] = Field(default_factory=list)

    @property
    def dangerous_errors(self) -> list[TrustCaseResult]:
        """Cases where the harness was too lenient (see ``is_dangerous_error``)."""
        return [case for case in self.cases if case.is_dangerous_error]


def _as_draft(case: TrustCase) -> ResumeDraft:
    """Wrap a single labeled claim as the one-bullet draft the harness scores.

    The harness's interface is draft-shaped, and the alternative — evaluating
    whole multi-claim drafts — makes a wrong verdict unattributable to a
    specific judgment and makes the label something two reviewers would
    disagree about.
    """
    return ResumeDraft(
        summary="", sections=[ResumeSection(heading="Experience", bullets=[case.claim])]
    )


def _as_evidence(case: TrustCase) -> EvidenceSet:
    """Wrap a case's evidence strings as the retrieval output the harness expects."""
    return EvidenceSet(
        user_id=EVAL_USER_ID,
        query=case.claim,
        chunks=[
            EvidenceChunk(
                chunk_id=f"{case.case_id}-{index}",
                document_id=case.case_id,
                user_id=EVAL_USER_ID,
                text=text,
            )
            for index, text in enumerate(case.evidence)
        ],
    )


def _dominant_status(report: TrustReport) -> ClaimStatus:
    """The single verdict a report implies: its worst claim.

    An empty report (the harness extracted no claims at all) counts as
    UNSUPPORTED, not as a skipped case — "I found nothing to verify" about a
    draft that plainly states something is a failure of the harness, and
    silently dropping those cases would hide it. This is exactly what the
    offline ``test`` provider does, which is why its Trust score is always 0.
    """
    if not report.claims:
        return ClaimStatus.UNSUPPORTED
    return min(report.claims, key=lambda claim: _SEVERITY.index(claim.status)).status


async def evaluate_trust(agent: SupportsTrustRun, cases: list[TrustCase]) -> TrustEvalReport:
    """Run every labeled claim through the Trust Harness and score its verdicts."""
    results: list[TrustCaseResult] = []
    for case in cases:
        report = await agent.run(draft=_as_draft(case), evidence=_as_evidence(case))
        results.append(
            TrustCaseResult(
                case_id=case.case_id,
                claim=case.claim,
                expected=case.expected_status,
                predicted=_dominant_status(report),
                note=case.note,
            )
        )

    metrics = classification_metrics(
        [case.expected.value for case in results],
        [case.predicted.value for case in results],
    )
    logger.info(
        "trust evaluation finished",
        extra={
            "cases": metrics.cases,
            "accuracy": round(metrics.accuracy, 3),
            "macro_f1": round(metrics.macro_f1, 3),
            "dangerous_errors": sum(1 for case in results if case.is_dangerous_error),
        },
    )
    return TrustEvalReport(metrics=metrics, cases=results)
