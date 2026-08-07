"""Unit tests for the browsable run-artifact directory writer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trustresume.export.artifacts import _slugify, _user_dir_name, write_run_artifacts
from trustresume.models import (
    ATSReport,
    ClaimCategory,
    ClaimStatus,
    JobDescription,
    ModelUsage,
    ResumeDraft,
    ResumeSection,
    RunUsage,
    TrustReport,
    VerifiedClaim,
    WorkflowState,
)


def _state(*, passed: bool = True, job: JobDescription | None = None) -> WorkflowState:
    draft = ResumeDraft(
        summary="Backend engineer.",
        sections=[ResumeSection(heading="Experience", bullets=["Built pipelines on AWS."])],
    )
    trust = TrustReport(
        claims=[
            VerifiedClaim(
                text="Built pipelines on AWS.",
                category=ClaimCategory.EXPERIENCE,
                status=ClaimStatus.SUPPORTED if passed else ClaimStatus.UNSUPPORTED,
                evidence_chunk_ids=["chunk-1"],
            )
        ],
        score=100.0 if passed else 0.0,
    )
    ats = ATSReport(
        score=100.0 if passed else 50.0,
        matched_keywords=["aws"],
        missing_keywords=[] if passed else ["kubernetes"],
    )
    return WorkflowState(
        user_id="ada",
        job=job or JobDescription(raw_text="Senior Backend Engineer", title="Senior Backend"),
        drafts=[draft],
        trust_reports=[trust],
        ats_reports=[ats],
        usage=RunUsage(
            models=[ModelUsage(model="m1", input_tokens=1000, output_tokens=200, calls=4)],
            total_duration_ms=8200.0,
            cost_usd=0.0123,
        ),
    )


def _write(tmp_path: Path, state: WorkflowState, **kwargs: object) -> Path:
    assert state.current_trust is not None and state.current_ats is not None
    return write_run_artifacts(
        tmp_path,
        state=state,
        resume_id="abcdef1234567890",
        trust=state.current_trust,
        ats=state.current_ats,
        markdown="# Resume\n",
        pdf=b"%PDF-1.4 fake",
        **kwargs,  # type: ignore[arg-type]
    )


# --- slugging --------------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Senior Backend Engineer", "senior-backend-engineer"),
        # Job titles are LLM output: path separators must never survive.
        ("../../etc/passwd", "etc-passwd"),
        ("Engineer\n\x00Nul", "engineer-nul"),
        ("!!!", "untitled"),
        ("", "untitled"),
        (None, "untitled"),
    ],
)
def test_slugify_sanitizesUntrustedTitles(title: str | None, expected: str) -> None:
    assert _slugify(title) == expected


def test_slugify_longTitle_truncated() -> None:
    assert len(_slugify("word " * 50)) <= 40


# --- directory layout ------------------------------------------------------


def test_writeRunArtifacts_createsAllFilesUnderUserScopedDirectory(tmp_path: Path) -> None:
    run_dir = _write(tmp_path, _state())

    assert run_dir.parent.name.startswith("ada-")  # scoped by user (ADR-0014)
    assert run_dir.name.endswith("-senior-backend-abcdef12")
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "evaluation.json",
        "evaluation.md",
        "job.md",
        "resume.md",
        "resume.pdf",
    ]
    assert (run_dir / "resume.md").read_text(encoding="utf-8") == "# Resume\n"
    assert (run_dir / "resume.pdf").read_bytes() == b"%PDF-1.4 fake"


def test_writeRunArtifacts_evaluationMarkdown_leadsWithTheVerdict(tmp_path: Path) -> None:
    run_dir = _write(tmp_path, _state(passed=True))

    text = (run_dir / "evaluation.md").read_text(encoding="utf-8")
    assert "**PASSED**" in text
    assert "Trust 100/90" in text
    assert "SUPPORTED" in text
    assert "4 LLM calls" in text
    assert "$0.0123" in text


def test_writeRunArtifacts_failedRun_recordsWhyAndWhatToFix(tmp_path: Path) -> None:
    run_dir = _write(
        tmp_path,
        _state(passed=False),
        rejection_reason="Trust score 0 (needs >= 90).",
        improvement_suggestions="Remove the unsupported claim.",
    )

    text = (run_dir / "evaluation.md").read_text(encoding="utf-8")
    assert "**DID NOT PASS**" in text
    assert "Trust score 0 (needs >= 90)." in text
    assert "Remove the unsupported claim." in text
    assert "flagged as unsupported factual assertions" in text
    assert "kubernetes" in text  # missing ATS keyword


def test_writeRunArtifacts_json_carriesTheAuditTrailMarkdownCannotShow(tmp_path: Path) -> None:
    """evidence_chunk_ids and the per-model token split justify the 2nd file."""
    run_dir = _write(tmp_path, _state())

    payload = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert payload["resume_id"] == "abcdef1234567890"
    assert payload["trust"]["claims"][0]["evidence_chunk_ids"] == ["chunk-1"]
    assert payload["usage"]["models"][0]["input_tokens"] == 1000
    assert payload["gate"]["min_trust_score"] == 90.0


def test_writeRunArtifacts_jobMarkdown_capturesThePostingForContext(tmp_path: Path) -> None:
    job = JobDescription(
        raw_text="Senior Backend Engineer at Acme",
        title="Senior Backend Engineer",
        company="Acme",
        required_skills=["python"],
        keywords=["aws"],
    )
    run_dir = _write(tmp_path, _state(job=job))

    text = (run_dir / "job.md").read_text(encoding="utf-8")
    assert "**Company:** Acme" in text
    assert "- python" in text
    assert "Senior Backend Engineer at Acme" in text


def test_writeRunArtifacts_noJob_stillWritesAUsableJobFile(tmp_path: Path) -> None:
    state = _state()
    state.job = None
    run_dir = _write(tmp_path, state)

    assert "No job description" in (run_dir / "job.md").read_text(encoding="utf-8")


def test_writeRunArtifacts_noUsage_omitsTheCostSection(tmp_path: Path) -> None:
    state = _state()
    state.usage = None
    run_dir = _write(tmp_path, state)

    assert "## Run cost" not in (run_dir / "evaluation.md").read_text(encoding="utf-8")


def test_writeRunArtifacts_unpricedModel_saysUnknownRatherThanZero(tmp_path: Path) -> None:
    state = _state()
    assert state.usage is not None
    state.usage.cost_usd = None
    run_dir = _write(tmp_path, state)

    text = (run_dir / "evaluation.md").read_text(encoding="utf-8")
    assert "unknown (no price configured" in text
    assert "$" not in text


def test_writeRunArtifacts_claimsOrderedWorstFirst(tmp_path: Path) -> None:
    """A reviewer should hit the problems before the fine claims."""
    state = _state()
    trust = TrustReport(
        claims=[
            VerifiedClaim(text="fine", status=ClaimStatus.SUPPORTED),
            VerifiedClaim(text="fabricated", status=ClaimStatus.UNSUPPORTED),
            VerifiedClaim(text="inflated", status=ClaimStatus.PARTIALLY_SUPPORTED),
        ],
        score=50.0,
    )
    state.trust_reports = [trust]

    run_dir = _write(tmp_path, state)

    text = (run_dir / "evaluation.md").read_text(encoding="utf-8")
    assert text.index("fabricated") < text.index("inflated") < text.index("fine")


def test_writeRunArtifacts_emptyClaimList_saysSoInsteadOfRenderingNothing(tmp_path: Path) -> None:
    state = _state()
    state.trust_reports = [TrustReport(claims=[], score=0.0)]

    run_dir = _write(tmp_path, state)

    assert "No claims were extracted" in (run_dir / "evaluation.md").read_text(encoding="utf-8")


# --- directory naming fallback --------------------------------------------


def test_writeRunArtifacts_noTitle_namesDirectoryFromCompanyThenPosting(tmp_path: Path) -> None:
    """A folder of '…-untitled-3375c09f' directories is useless to scan."""
    no_title = JobDescription(raw_text="Staff Platform Engineer wanted", company="Acme Corp")
    assert _write(tmp_path, _state(job=no_title)).name.endswith("-acme-corp-abcdef12")

    nothing_extracted = JobDescription(raw_text="Staff Platform Engineer wanted, remote")
    run_dir = _write(tmp_path, _state(job=nothing_extracted))
    assert "staff-platform-engineer-wanted" in run_dir.name


def test_writeRunArtifacts_unnameableJob_fallsBackToUntitled(tmp_path: Path) -> None:
    run_dir = _write(tmp_path, _state(job=JobDescription(raw_text="の")))
    assert run_dir.name.endswith("-untitled-abcdef12")


def test_writeRunArtifacts_subSecondRun_reportsMillisecondsNotZeroSeconds(
    tmp_path: Path,
) -> None:
    """'0.0s' for a 44ms run reads like a broken measurement."""
    state = _state()
    assert state.usage is not None
    state.usage.total_duration_ms = 44.3

    run_dir = _write(tmp_path, state)

    assert "44ms wall clock" in (run_dir / "evaluation.md").read_text(encoding="utf-8")


def test_userDirName_distinctIdsNeverShareADirectory() -> None:
    """Slugging alone would merge these, and they are different accounts.

    Case matters too: macOS filesystems are case-insensitive by default, so
    'Ada' and 'ada' would collide there even if stored verbatim. The hash
    suffix is what makes the mapping injective on any filesystem.
    """
    names = {_user_dir_name(uid) for uid in ("ada", "Ada", "a.b", "a-b", "a_b")}
    assert len(names) == 5
    assert all(name.startswith(("ada-", "a-b-")) for name in names)  # still readable
