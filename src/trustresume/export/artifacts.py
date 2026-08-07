"""Write one generation's résumé and evaluation to a browsable directory.

SQLite already holds everything (draft, exports, scores, usage), and the API
can serve it — but neither is *browsable*. Opening a folder and reading the
last five runs side by side is how you actually review output, compare a
rewrite against its predecessor, or hand a draft to someone who doesn't have
the app running.

This is a **convenience view, not a source of truth**: the database write in
``TrustResumeApp._persist`` happens first and is authoritative. A failure here
(full disk, read-only mount, a path the container can't write) must never
turn a successful generation into an error — the caller swallows ``OSError``
and logs it.

Layout, one directory per run::

    <output_dir>/<user_id>/<timestamp>-<job-slug>-<short-resume-id>/
        resume.md          the draft, ready to read or paste
        resume.pdf         the same draft, rendered
        evaluation.md      human-readable verdict: scores, flagged claims, cost
        evaluation.json    full detail the Markdown can't usefully render
        job.md             the posting this was written against, for context

Scoped by ``user_id`` because two users can generate concurrently (ADR-0014);
a flat directory would interleave two people's résumés and risk name
collisions.

Added post-port; no equivalent in the original.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from trustresume.models import ATSReport, TrustReport, WorkflowState

#: Longest job slug allowed in a directory name. Long enough to stay
#: recognizable, short enough that the full path clears filesystem limits once
#: the timestamp, id, and filename are appended.
_MAX_SLUG_CHARS = 40

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9]+")

#: Used when nothing about a job sanitizes to a usable name (a posting written
#: entirely in non-Latin script, say) — a directory named for its timestamp and
#: id is still usable, just less recognizable at a glance.
_UNTITLED_SLUG = "untitled"

#: How much raw posting text to fall back on when no title was extracted.
#: Trimmed further by ``_MAX_SLUG_CHARS``; this just bounds the work.
_RAW_TEXT_SLUG_CHARS = 80


def _slugify(text: str | None) -> str:
    """A filesystem-safe fragment of ``text``.

    Job titles come from LLM output, which is untrusted text: it can contain
    path separators, newlines, null bytes, or run to hundreds of characters.
    Allow-listing alphanumerics (rather than blocking known-bad ones) is the
    same reasoning as ``server.py``'s user-id pattern — the safe set is small
    and knowable, the unsafe set isn't.
    """
    if not text:
        return _UNTITLED_SLUG
    slug = _UNSAFE_CHARS.sub("-", text).strip("-").lower()
    return slug[:_MAX_SLUG_CHARS].strip("-") or _UNTITLED_SLUG


def _run_slug(state: WorkflowState) -> str:
    """The recognizable part of a run directory's name.

    Falls back title → company → the posting's opening words, mirroring
    ``app_service._job_summary``'s ladder. Worth the fallback because the
    whole feature is about *browsing*: a folder of ``…-untitled-3375c09f``
    directories is technically correct and useless to scan, and a job with no
    extracted title is common (the offline provider never emits one).
    """
    job = state.job
    if job is None:
        return _UNTITLED_SLUG
    for candidate in (job.title, job.company, job.raw_text[:_RAW_TEXT_SLUG_CHARS]):
        slug = _slugify(candidate)
        if slug != _UNTITLED_SLUG:
            return slug
    return _UNTITLED_SLUG


def _format_duration(milliseconds: float) -> str:
    """Wall clock a human can read: ``840ms`` under a second, ``8.2s`` above.

    ``0.0s`` for a run that took 44ms is technically true and reads like a
    measurement failure.
    """
    if milliseconds < 1000:
        return f"{milliseconds:.0f}ms"
    return f"{milliseconds / 1000:.1f}s"


def _user_dir_name(user_id: str) -> str:
    """A directory name that is readable *and* unique per user id.

    Slugifying alone is not safe here: it lowercases and collapses every
    non-alphanumeric run, so ``Ada``/``ada`` and ``a.b``/``a-b`` would all
    land in one folder — and those are genuinely different accounts (SQLite
    ids are binary-compared, and the API's id pattern admits case, ``.``,
    ``_`` and ``-``). Interleaving two people's résumés is exactly the
    collision this per-user scoping exists to prevent.

    Case alone can't fix it either, because macOS filesystems are
    case-insensitive by default — ``Ada/`` and ``ada/`` are the same directory
    there and different on Linux. So the readable slug always carries a short
    hash of the *exact* id, which restores uniqueness on any filesystem at the
    cost of eight characters.
    """
    digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:8]
    return f"{_slugify(user_id)}-{digest}"


def _format_claims(trust: TrustReport) -> str:
    """The Trust Harness verdict as a Markdown list, worst claims first."""
    if not trust.claims:
        return "_No claims were extracted from this draft._\n"
    order = {"UNSUPPORTED": 0, "PARTIALLY_SUPPORTED": 1, "SUPPORTED": 2}
    ranked = sorted(trust.claims, key=lambda claim: order.get(claim.status.value, 3))
    lines = [f"- **{c.status.value}** ({c.category.value}) — {c.text}" for c in ranked]
    return "\n".join(lines) + "\n"


def _evaluation_markdown(
    *,
    state: WorkflowState,
    trust: TrustReport,
    ats: ATSReport,
    rejection_reason: str | None,
    improvement_suggestions: str | None,
) -> str:
    """The reviewer-facing summary: verdict first, then why, then what it cost."""
    verdict = "PASSED" if state.passed else "DID NOT PASS"
    gate = state.gate
    parts = [
        "# Evaluation\n\n",
        f"**{verdict}** — Trust {trust.score:.0f}/{gate.min_trust_score:.0f} · "
        f"ATS {ats.score:.0f}/{gate.min_ats_score:.0f} · "
        f"iteration {state.iteration} of {gate.max_iterations}\n",
    ]
    if rejection_reason:
        parts.append(f"**Why it failed:** {rejection_reason}\n")
    if state.is_exhausted and not state.passed:
        parts.append(
            "_Hit the rewrite cap without passing — this is the last draft, "
            "exported with its real scores rather than discarded._\n"
        )

    parts.append("\n## Trust Harness\n\n")
    parts.append(_format_claims(trust))
    if trust.hallucinations:
        parts.append(
            f"\n**{len(trust.hallucinations)} claim(s) flagged as unsupported factual "
            "assertions — the ones to fix first.**\n"
        )

    parts.append("\n## ATS keyword coverage\n\n")
    parts.append(f"- Matched: {', '.join(ats.matched_keywords) or '(none)'}\n")
    parts.append(f"- Missing: {', '.join(ats.missing_keywords) or '(none)'}\n")
    if ats.notes:
        parts.append(f"- Notes: {ats.notes}\n")

    if improvement_suggestions:
        parts.append("\n## Suggested improvements\n\n")
        parts.append(f"```\n{improvement_suggestions}\n```\n")

    if state.usage:
        usage = state.usage
        cost = "unknown (no price configured for this model)"
        if usage.cost_usd is not None:
            cost = f"${usage.cost_usd:.4f}"
        parts.append("\n## Run cost\n\n")
        parts.append(
            f"- {usage.llm_calls} LLM calls · {usage.total_tokens:,} tokens "
            f"({usage.input_tokens:,} in / {usage.output_tokens:,} out)\n"
        )
        parts.append(f"- {_format_duration(usage.total_duration_ms)} wall clock · {cost}\n")
    return "".join(parts)


def _evaluation_json(
    *,
    state: WorkflowState,
    trust: TrustReport,
    ats: ATSReport,
    resume_id: str,
    rejection_reason: str | None,
    improvement_suggestions: str | None,
) -> str:
    """Everything the Markdown summary can't render usefully.

    Carries per-claim ``evidence_chunk_ids`` (which verdict rested on which
    retrieved chunk — the audit trail), the per-model token split, and
    per-node timings. That's the difference between "readable" and
    "diffable/queryable", which is why both files exist.
    """
    payload = {
        "resume_id": resume_id,
        "user_id": state.user_id,
        "job_id": state.job_id,
        "iteration": state.iteration,
        "passed": state.passed,
        "exhausted": state.is_exhausted,
        "gate": state.gate.model_dump(),
        "trust": trust.model_dump(mode="json"),
        "ats": ats.model_dump(mode="json"),
        "rejection_reason": rejection_reason,
        "improvement_suggestions": improvement_suggestions,
        "usage": state.usage.model_dump(mode="json") if state.usage else None,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _job_markdown(state: WorkflowState) -> str:
    """The posting this draft was written against, for review context."""
    job = state.job
    if job is None:
        return "# Target job\n\n_No job description was captured for this run._\n"
    lines = ["# Target job\n"]
    if job.title:
        lines.append(f"**Title:** {job.title}\n")
    if job.company:
        lines.append(f"**Company:** {job.company}\n")
    lines.append(f"**Seniority:** {job.seniority.value}\n")
    for label, values in (
        ("Required skills", job.required_skills),
        ("Preferred skills", job.preferred_skills),
        ("Responsibilities", job.responsibilities),
        ("ATS keywords", job.keywords),
    ):
        if values:
            lines.append(f"\n## {label}\n")
            lines.extend(f"- {value}\n" for value in values)
    lines.append("\n## Original posting\n")
    lines.append(f"```\n{job.raw_text}\n```\n")
    return "".join(lines)


def write_run_artifacts(
    output_dir: Path,
    *,
    state: WorkflowState,
    resume_id: str,
    trust: TrustReport,
    ats: ATSReport,
    markdown: str,
    pdf: bytes,
    rejection_reason: str | None = None,
    improvement_suggestions: str | None = None,
) -> Path:
    """Write one run's files and return the directory created.

    ``markdown``/``pdf`` are passed in rather than re-rendered here: the caller
    already rendered them for the database write, and rendering twice could
    (however unlikely) produce a file that differs from the stored bytes.

    Raises ``OSError`` on any filesystem problem — the caller decides whether
    that's fatal. In ``_persist`` it isn't: the database already has
    everything.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = (
        output_dir
        / _user_dir_name(state.user_id)
        / f"{timestamp}-{_run_slug(state)}-{resume_id[:8]}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "resume.md").write_text(markdown, encoding="utf-8")
    (run_dir / "resume.pdf").write_bytes(pdf)
    (run_dir / "evaluation.md").write_text(
        _evaluation_markdown(
            state=state,
            trust=trust,
            ats=ats,
            rejection_reason=rejection_reason,
            improvement_suggestions=improvement_suggestions,
        ),
        encoding="utf-8",
    )
    (run_dir / "evaluation.json").write_text(
        _evaluation_json(
            state=state,
            trust=trust,
            ats=ats,
            resume_id=resume_id,
            rejection_reason=rejection_reason,
            improvement_suggestions=improvement_suggestions,
        ),
        encoding="utf-8",
    )
    (run_dir / "job.md").write_text(_job_markdown(state), encoding="utf-8")
    return run_dir
