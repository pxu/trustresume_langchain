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
    NodeTiming,
    NodeUsage,
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
        "metrics.json",
        "resume.md",
        "resume.pdf",
    ]
    assert (run_dir / "resume.md").read_text(encoding="utf-8") == "# Resume\n"
    assert (run_dir / "resume.pdf").read_bytes() == b"%PDF-1.4 fake"


def test_writeRunArtifacts_evaluationMarkdown_leadsWithTheVerdict(tmp_path: Path) -> None:
    run_dir = _write(tmp_path, _state(passed=True))

    text = (run_dir / "evaluation.md").read_text(encoding="utf-8")
    assert "**Verdict:** PASSED" in text
    assert "| Trust | 100.0 | 90 | yes |" in text
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
    assert "**Verdict:** DID NOT PASS" in text
    assert "Trust score 0 (needs >= 90)." in text
    assert "Remove the unsupported claim." in text
    assert "Flagged as unsupported: 1" in text
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


# --- iteration history ------------------------------------------------


def test_writeRunArtifacts_singleDraft_omitsIterationHistory(tmp_path: Path) -> None:
    """Nothing to trend with one draft — the table would be noise."""
    run_dir = _write(tmp_path, _state())
    assert "## Iteration history" not in (run_dir / "evaluation.md").read_text(encoding="utf-8")


def test_writeRunArtifacts_multipleDrafts_markdownShowsScoreTrajectory(tmp_path: Path) -> None:
    state = _state(passed=False)
    early_trust = TrustReport(claims=[], score=40.0)
    early_ats = ATSReport(score=20.0, matched_keywords=[], missing_keywords=["kubernetes"])
    state.drafts = [state.drafts[0], state.drafts[0]]
    state.trust_reports = [early_trust, state.trust_reports[0]]
    state.ats_reports = [early_ats, state.ats_reports[0]]

    run_dir = _write(tmp_path, state)

    text = (run_dir / "evaluation.md").read_text(encoding="utf-8")
    assert "## Iteration history" in text
    assert "| 0 | 40 | 20 | no |" in text
    assert "| 1 (exported) | 0 | 50 | no |" in text


def test_writeRunArtifacts_json_carriesEveryIterationsDraftAndReports(tmp_path: Path) -> None:
    """Without this, whether the exported draft beat earlier ones is unrecoverable."""
    state = _state(passed=False)
    early_trust = TrustReport(claims=[], score=40.0)
    early_ats = ATSReport(score=20.0, matched_keywords=[], missing_keywords=["kubernetes"])
    state.drafts = [state.drafts[0], state.drafts[0]]
    state.trust_reports = [early_trust, state.trust_reports[0]]
    state.ats_reports = [early_ats, state.ats_reports[0]]

    run_dir = _write(tmp_path, state)

    payload = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert [entry["iteration"] for entry in payload["iterations"]] == [0, 1]
    assert payload["iterations"][0]["trust"]["score"] == 40.0
    assert payload["iterations"][1]["trust"]["score"] == 0.0
    assert payload["iterations"][0]["draft"]["summary"] == "Backend engineer."


# --- metrics.json -----------------------------------------------------


def test_writeRunArtifacts_noUsage_skipsMetricsFileEntirely(tmp_path: Path) -> None:
    """An all-zero metrics.json would look like a real (if cheap) measurement."""
    state = _state()
    state.usage = None

    run_dir = _write(tmp_path, state)

    assert not (run_dir / "metrics.json").exists()


def test_writeRunArtifacts_metricsJson_attributesStepsToTheirIteration(tmp_path: Path) -> None:
    """Two iterations worth of node executions, split back out per step."""
    state = _state(passed=False)
    state.usage = RunUsage(
        models=[ModelUsage(model="m1", input_tokens=300, output_tokens=90, calls=3)],
        timings=[
            NodeTiming(node="write_resume", duration_ms=10.0),
            NodeTiming(node="score_trust", duration_ms=5.0),
            NodeTiming(node="score_ats", duration_ms=0.1),
            NodeTiming(node="prepare_rewrite", duration_ms=0.2),
            NodeTiming(node="write_resume", duration_ms=12.0),
            NodeTiming(node="score_trust", duration_ms=6.0),
        ],
        node_calls=[
            NodeUsage(node="write_resume", model="m1", input_tokens=100, output_tokens=30),
            NodeUsage(node="score_trust", model="m1", input_tokens=100, output_tokens=30),
            NodeUsage(node="write_resume", model="m1", input_tokens=100, output_tokens=30),
            NodeUsage(node="score_trust", model="m1", input_tokens=100, output_tokens=30),
        ],
        total_duration_ms=8200.0,
        cost_usd=0.0123,
    )

    run_dir = _write(tmp_path, state)

    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert payload["totals"]["llm_requests"] == 3
    assert payload["totals"]["estimated_cost_usd"] == 0.0123

    steps = payload["steps"]
    assert [s["step"] for s in steps] == [
        "write_resume",
        "score_trust",
        "score_ats",
        "prepare_rewrite",
        "write_resume",
        "score_trust",
    ]
    # score_ats/prepare_rewrite make no LLM call: zero tokens, not missing.
    assert steps[2]["input_tokens"] == 0
    assert steps[2]["requests"] == 0
    # prepare_rewrite is where the orchestrator increments iteration, so
    # everything from index 4 on belongs to iteration 1.
    assert [s["iteration"] for s in steps] == [0, 0, 0, 0, 1, 1]
    assert steps[0]["input_tokens"] == 100
    assert steps[4]["input_tokens"] == 100


def test_writeRunArtifacts_metricsJson_missingNodeCall_notFalselyZeroed(tmp_path: Path) -> None:
    """A node with no matching NodeUsage entry still gets a step, just no tokens."""
    state = _state()
    state.usage = RunUsage(
        models=[ModelUsage(model="m1", input_tokens=0, output_tokens=0, calls=0)],
        timings=[NodeTiming(node="retrieve_evidence", duration_ms=1.0)],
        total_duration_ms=1.0,
    )

    run_dir = _write(tmp_path, state)

    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert payload["steps"] == [
        {
            "step": "retrieve_evidence",
            "iteration": 0,
            "duration_ms": 1.0,
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost_usd": 0.0,
        }
    ]


def test_userDirName_distinctIdsNeverShareADirectory() -> None:
    """Slugging alone would merge these, and they are different accounts.

    Case matters too: macOS filesystems are case-insensitive by default, so
    'Ada' and 'ada' would collide there even if stored verbatim. The hash
    suffix is what makes the mapping injective on any filesystem.
    """
    names = {_user_dir_name(uid) for uid in ("ada", "Ada", "a.b", "a-b", "a_b")}
    assert len(names) == 5
    assert all(name.startswith(("ada-", "a-b-")) for name in names)  # still readable
