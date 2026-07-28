"""Unit tests for the Trust Harness and ATS schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trustresume.models import (
    ATSReport,
    ClaimCategory,
    ClaimStatus,
    TrustReport,
    VerifiedClaim,
)


def _claim(status: ClaimStatus, category: ClaimCategory = ClaimCategory.SKILL) -> VerifiedClaim:
    return VerifiedClaim(text="Knows Kubernetes", status=status, category=category)


def test_verifiedClaim_unsupportedSkill_isHallucination() -> None:
    assert _claim(ClaimStatus.UNSUPPORTED, ClaimCategory.SKILL).is_hallucination is True


def test_verifiedClaim_unsupportedOther_isNotHallucination() -> None:
    # A stylistic/unclassified unsupported claim is not flagged as a hallucination.
    assert _claim(ClaimStatus.UNSUPPORTED, ClaimCategory.OTHER).is_hallucination is False


def test_verifiedClaim_supported_isNotHallucination() -> None:
    assert _claim(ClaimStatus.SUPPORTED, ClaimCategory.SKILL).is_hallucination is False


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ([], 0.0),
        ([ClaimStatus.SUPPORTED], 100.0),
        ([ClaimStatus.UNSUPPORTED], 0.0),
        ([ClaimStatus.PARTIALLY_SUPPORTED], 50.0),
        ([ClaimStatus.SUPPORTED, ClaimStatus.UNSUPPORTED], 50.0),
        ([ClaimStatus.SUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED, ClaimStatus.UNSUPPORTED], 50.0),
    ],
)
def test_trustReport_computeScore_rubric(statuses: list[ClaimStatus], expected: float) -> None:
    claims = [_claim(s) for s in statuses]
    assert TrustReport.compute_score(claims) == pytest.approx(expected)


def test_trustReport_fromClaims_setsScoreAndProperties() -> None:
    claims = [
        _claim(ClaimStatus.SUPPORTED),
        _claim(ClaimStatus.UNSUPPORTED, ClaimCategory.CERTIFICATION),
        _claim(ClaimStatus.UNSUPPORTED, ClaimCategory.OTHER),
    ]
    report = TrustReport.from_claims(claims, iteration=1)

    assert report.score == pytest.approx(round(100.0 / 3, 2))
    assert report.iteration == 1
    assert report.supported_fraction == pytest.approx(1 / 3)
    # Only the UNSUPPORTED certification is a hallucination, not the OTHER claim.
    assert len(report.hallucinations) == 1
    assert report.hallucinations[0].category is ClaimCategory.CERTIFICATION


def test_trustReport_emptyClaims_supportedFractionZero() -> None:
    assert TrustReport(claims=[], score=0.0).supported_fraction == 0.0


def test_trustReport_scoreOutOfRange_raises() -> None:
    with pytest.raises(ValidationError):
        TrustReport(claims=[], score=101.0)


def test_atsReport_matchedMissingOverlap_raises() -> None:
    with pytest.raises(ValidationError):
        ATSReport(score=80.0, matched_keywords=["python"], missing_keywords=["python"])


def test_atsReport_valid() -> None:
    report = ATSReport(
        score=88.0,
        matched_keywords=["python", "aws"],
        missing_keywords=["kubernetes"],
    )
    assert report.score == pytest.approx(88.0)
    assert report.iteration == 0
