"""Building a semantic-search query string from a structured job description.

Split out of ``agents/retrieval_agent.py`` so the term list it concatenates
has one home other callers (evaluation notebooks, future retrieval-quality
tooling) can reuse without reaching into the agent's internals.

Milestone M2 (storage + retrieval).
"""

from __future__ import annotations

from trustresume.models import JobDescription


def query_terms(job: JobDescription) -> list[str]:
    """The ordered terms :func:`build_query` joins into one query string.

    Skills and keywords carry the strongest signal for matching career
    evidence; the title anchors seniority/domain. Exposed separately from
    :func:`build_query` so callers that want per-term granularity (e.g.
    evaluating retrieval quality one skill at a time) can reuse the same
    term list instead of re-deriving it.
    """
    return [
        *([job.title] if job.title else []),
        *job.required_skills,
        *job.preferred_skills,
        *job.keywords,
    ]


def build_query(job: JobDescription) -> str:
    """Compose a retrieval query from the most job-relevant fields.

    Falls back to the raw posting text when nothing structured was
    extracted.
    """
    query = " ".join(query_terms(job)).strip()
    return query or job.raw_text


def per_skill_queries(job: JobDescription) -> list[str]:
    """`required_skills + preferred_skills`, deduped — one query per skill.

    Not a different scheme from :func:`build_query` — the same two lists it
    already concatenates into one query string, evaluated one entry at a
    time instead. Useful for callers that want a query per individual skill
    (e.g. retrieval-quality evaluation) rather than one combined query.
    `keywords` is deliberately excluded: it's a separately-worded, deduped
    list (e.g. "REST API" vs. `required_skills`' "REST API design"), so
    treating its entries as more per-skill queries would introduce terms not
    present in `required_skills`/`preferred_skills`.
    """
    return list(dict.fromkeys(job.required_skills + job.preferred_skills))
