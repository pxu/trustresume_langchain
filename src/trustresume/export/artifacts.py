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
        metrics.json       per-orchestrator-step tokens/cost/duration

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

from trustresume.models import ATSReport, NodeUsage, RunUsage, TrustReport, WorkflowState

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


def _iteration_history_markdown(state: WorkflowState) -> str:
    """A per-iteration score table — how the quality loop actually got here.

    Only worth showing once there was a rewrite: a single-draft run has
    nothing to trend. ``drafts``/``trust_reports``/``ats_reports`` are kept
    parallel by the orchestrator (see ``WorkflowState``), so zipping them by
    index is safe. The exported row is ``state.final_index`` — the loop never
    stops early on a pass, so it is routinely *not* the last row: it is
    whichever iteration ranks best by (passed the gate, ATS score), which the
    table makes auditable.
    """
    if len(state.drafts) <= 1:
        return ""
    lines = [
        "\n## Iteration history\n\n",
        "| Iteration | Trust | ATS | Passed |\n",
        "|---|---|---|---|\n",
    ]
    exported = state.final_index
    for index in range(len(state.drafts)):
        trust_i = state.trust_reports[index] if index < len(state.trust_reports) else None
        ats_i = state.ats_reports[index] if index < len(state.ats_reports) else None
        passed = trust_i is not None and ats_i is not None and state.gate.passes(trust_i, ats_i)
        trust_str = f"{trust_i.score:.0f}" if trust_i else "—"
        ats_str = f"{ats_i.score:.0f}" if ats_i else "—"
        marker = " (exported)" if index == exported else ""
        verdict = "yes" if passed else "no"
        lines.append(f"| {index}{marker} | {trust_str} | {ats_str} | {verdict} |\n")
    return "".join(lines)


def _slowest_step(usage: RunUsage) -> tuple[str, float] | None:
    """The single node execution (not node-total) that took the longest.

    Deliberately the slowest *execution*, not the slowest node-by-total: a
    node that runs four times at 2s each is a different problem (it's slow
    every time) than one that runs once at 8s (something about that one call
    was unusual) — collapsing to totals would conflate the two.
    """
    if not usage.timings:
        return None
    slowest = max(usage.timings, key=lambda timing: timing.duration_ms)
    return slowest.node, slowest.duration_ms


def _evaluation_markdown(
    *,
    state: WorkflowState,
    trust: TrustReport,
    ats: ATSReport,
    rejection_reason: str | None,
    improvement_suggestions: str | None,
) -> str:
    """The reviewer-facing summary: verdict first, then why, then what it cost."""
    verdict = "PASSED" if state.final_passed else "DID NOT PASS"
    gate = state.gate
    job_title = state.job.title if state.job and state.job.title else "(no title extracted)"
    parts = [
        "# Evaluation\n\n",
        f"- **Verdict:** {verdict}\n",
        f"- **Job title:** {job_title}\n",
        f"- **Gate:** Trust ≥ {gate.min_trust_score:.0f} and ATS ≥ {gate.min_ats_score:.0f}\n",
        f"- **Iterations run:** {state.iteration} (cap {gate.max_iterations}; "
        f"{len(state.drafts)} draft(s) total)\n",
    ]
    if state.usage and state.usage.models:
        parts.append(f"- **Model(s):** {', '.join(m.model for m in state.usage.models)}\n")
    parts.append(
        f"\n## Scores (iteration {state.final_index})\n\n"
        "| Metric | Score | Threshold | Met |\n"
        "|---|---|---|---|\n"
        f"| Trust | {trust.score:.1f} | {gate.min_trust_score:.0f} | "
        f"{'yes' if trust.score >= gate.min_trust_score else '**no**'} |\n"
        f"| ATS | {ats.score:.1f} | {gate.min_ats_score:.0f} | "
        f"{'yes' if ats.score >= gate.min_ats_score else '**no**'} |\n"
    )
    if rejection_reason:
        parts.append(f"\n**Why it failed:** {rejection_reason}\n")
    if not state.final_passed:
        parts.append(
            "\n_None of this run's drafts passed the gate — this is the "
            "best-scoring one across all iterations (highest ATS, since none "
            "passed to rank by), exported with its real scores rather than "
            "discarded._\n"
        )
    parts.append(_iteration_history_markdown(state))

    parts.append("\n## Trust Harness\n\n")
    supported = sum(1 for c in trust.claims if c.status.value == "SUPPORTED")
    parts.append(f"- Claims extracted: {len(trust.claims)}\n")
    if trust.claims:
        parts.append(f"- Supported fraction: {supported / len(trust.claims):.2f}\n")
    parts.append(f"- Flagged as unsupported: {len(trust.hallucinations)}\n\n")
    parts.append(_format_claims(trust))

    matched = ", ".join(ats.matched_keywords) or "(none)"
    missing = ", ".join(ats.missing_keywords) or "(none)"
    parts.append("\n## ATS keyword coverage\n\n")
    parts.append(f"- Matched ({len(ats.matched_keywords)}): {matched}\n")
    parts.append(f"- Missing ({len(ats.missing_keywords)}): {missing}\n")
    if ats.notes:
        parts.append(f"- Notes: {ats.notes}\n")

    if improvement_suggestions:
        parts.append("\n## Rewrite feedback\n\n")
        parts.append(
            "Deterministic, built from the gate's own gap (`orchestration/feedback.py`) — "
            "no extra LLM call.\n\n"
        )
        parts.append(f"```\n{improvement_suggestions}\n```\n")

    if state.usage:
        usage = state.usage
        cost = "unknown (no price configured for this model)"
        if usage.cost_usd is not None:
            cost = f"${usage.cost_usd:.4f}"
        parts.append("\n## Cost & latency\n\n")
        parts.append(
            f"- {usage.llm_calls} LLM calls · {usage.total_tokens:,} tokens "
            f"({usage.input_tokens:,} in / {usage.output_tokens:,} out) · {cost}\n"
        )
        parts.append(f"- {_format_duration(usage.total_duration_ms)} wall clock\n")
        slowest = _slowest_step(usage)
        if slowest is not None:
            node, duration_ms = slowest
            parts.append(f"- Slowest step: {node} ({_format_duration(duration_ms)})\n")
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

    ``iterations`` additionally carries every draft + report the quality loop
    produced, not just the exported one — ``trust``/``ats`` above are the
    exported (best-scoring) draft's, kept as top-level fields so existing
    readers of this file don't need to know the history exists. Keeping the
    full history is what makes the best-of selection auditable: which
    iteration was exported (``final_index``) and why it beat the others is
    recoverable from the per-iteration scores here.
    """
    iterations = [
        {
            "iteration": index,
            "draft": draft.model_dump(mode="json"),
            "trust": (
                state.trust_reports[index].model_dump(mode="json")
                if index < len(state.trust_reports)
                else None
            ),
            "ats": (
                state.ats_reports[index].model_dump(mode="json")
                if index < len(state.ats_reports)
                else None
            ),
        }
        for index, draft in enumerate(state.drafts)
    ]
    payload = {
        "resume_id": resume_id,
        "user_id": state.user_id,
        "job_id": state.job_id,
        "iteration": state.iteration,
        "exported_iteration": state.final_index,
        "passed": state.final_passed,
        "gate": state.gate.model_dump(),
        "trust": trust.model_dump(mode="json"),
        "ats": ats.model_dump(mode="json"),
        "rejection_reason": rejection_reason,
        "improvement_suggestions": improvement_suggestions,
        "usage": state.usage.model_dump(mode="json") if state.usage else None,
        "iterations": iterations,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _steps_with_iteration(usage: RunUsage) -> list[dict[str, object]]:
    """Zip ``timings`` (one per node execution) with ``node_calls`` (one per LLM
    call), reconstructing which iteration of the quality loop each step ran in.

    Neither list carries an iteration number on its own — ``timings`` only
    knows node names and durations, ``node_calls`` only knows tokens — so this
    walks ``timings`` in execution order and counts how many times
    ``prepare_rewrite`` has already run: that node is the only place the
    orchestrator increments ``iteration`` (see ``orchestrator._prepare_rewrite``),
    so everything before the first one is iteration 0, and so on. Each node
    name gets its own FIFO queue of ``node_calls``, since a node with no LLM
    call this execution (``retrieve_evidence``, ``score_ats``,
    ``prepare_rewrite``) simply has an empty queue to pop from.
    """
    calls_by_node: dict[str, list[NodeUsage]] = {}
    for node_call in usage.node_calls:
        calls_by_node.setdefault(node_call.node, []).append(node_call)

    steps: list[dict[str, object]] = []
    iteration = 0
    for timing in usage.timings:
        pending = calls_by_node.get(timing.node, [])
        call: NodeUsage | None = pending.pop(0) if pending else None
        steps.append(
            {
                "step": timing.node,
                "iteration": iteration,
                "duration_ms": round(timing.duration_ms, 3),
                "requests": 1 if call else 0,
                "input_tokens": call.input_tokens if call else 0,
                "output_tokens": call.output_tokens if call else 0,
                "estimated_cost_usd": call.cost_usd if call else 0.0,
            }
        )
        if timing.node == "prepare_rewrite":
            iteration += 1
    return steps


def _metrics_json(state: WorkflowState) -> str | None:
    """Per-step tokens, latency, and cost — the audit trail for *where it went*.

    ``evaluation.json`` answers "what did the whole run cost"; this answers
    "which step of which iteration". Returns ``None`` when there's no usage to
    report (a state built without going through a real orchestrator run, e.g.
    directly in a test) — callers skip writing the file rather than write an
    all-zero one that would look like a measurement.
    """
    if state.usage is None:
        return None
    usage = state.usage
    payload = {
        "models": [model.model_dump(mode="json") for model in usage.models],
        "totals": {
            "llm_requests": usage.llm_calls,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "duration_ms": round(usage.total_duration_ms, 3),
            "estimated_cost_usd": usage.cost_usd,
        },
        "steps": _steps_with_iteration(usage),
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
    metrics = _metrics_json(state)
    if metrics is not None:
        (run_dir / "metrics.json").write_text(metrics, encoding="utf-8")
    return run_dir
