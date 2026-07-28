"""Reusable ATS evaluation logic: keyword coverage scoring.

Extracted from the ATS Evaluation agent so the scoring is a pure, reusable
function independent of any agent wrapper. Deterministic and reproducible — the
score is computed here, never self-reported by an LLM.

Milestone M6 (trust verification + evaluation).
"""

from __future__ import annotations

from trustresume.models import ATSReport, JobDescription, ResumeDraft


def draft_text(draft: ResumeDraft) -> str:
    """Flatten a draft to lowercased text for case-insensitive matching."""
    parts = [draft.summary]
    for section in draft.sections:
        parts.append(section.heading)
        parts.extend(section.bullets)
    return "\n".join(parts).lower()


def dedupe(items: list[str]) -> list[str]:
    """Order-preserving, case-insensitive dedupe."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def score_keywords(draft: ResumeDraft, job: JobDescription) -> ATSReport:
    """Score a draft's keyword coverage against a job description.

    Coverage is ``matched / total`` over the job's target keywords (falling
    back to required skills when no explicit keywords were extracted), scaled
    to 0–100. A job with no keywords scores 100 — nothing to match against, so
    the draft isn't penalized.
    """
    keywords = dedupe(job.keywords or job.required_skills)
    text = draft_text(draft)

    matched = [kw for kw in keywords if kw.lower() in text]
    missing = [kw for kw in keywords if kw.lower() not in text]
    score = 100.0 if not keywords else round(100.0 * len(matched) / len(keywords), 2)

    return ATSReport(
        score=score,
        matched_keywords=matched,
        missing_keywords=missing,
        iteration=draft.iteration,
    )


def required_skill_coverage(job: JobDescription, candidate_skills: list[str]) -> dict[str, object]:
    """Exact-token coverage of the job's required skills by a candidate's actual skills.

    Case-insensitive set intersection — no substring matching, so unlike
    ``score_keywords`` it can't spuriously match a short required skill (e.g.
    "SQL") inside an unrelated skill the candidate has (e.g. "PostgreSQL").

    This is a cross-check, not a replacement for ``score_keywords``:
    ``job.required_skills`` is written in full sentences for the Resume
    Writer, so exact-token matching against it is usually near 0% regardless
    of actual fit. Match against ``job.keywords`` for a meaningful score (see
    `docs/weekly-submissions/week3-baseline-model.md`'s "key finding").
    """
    required = {s.lower(): s for s in job.required_skills}
    actual = {s.lower() for s in candidate_skills}
    matched = [orig for key, orig in required.items() if key in actual]
    missing = [orig for key, orig in required.items() if key not in actual]
    coverage = len(matched) / len(required) if required else 1.0
    return {"coverage": coverage, "matched": matched, "missing": missing}
