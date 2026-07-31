"""Unit tests for ``api/app_service.py``'s ``_job_summary`` helper.

A pure function (title/company -> a short display label), tested directly
rather than only indirectly through ``create_job``/``update_job`` — every
branch (both known, title-only, company-only, neither) needs its own case.
"""

from __future__ import annotations

from trustresume.api.app_service import _job_summary
from trustresume.models import JobDescription


def test_jobSummary_titleAndCompany_combinesBoth() -> None:
    job = JobDescription(raw_text="x", title="Senior Engineer", company="Acme")
    assert _job_summary(job) == "Senior Engineer at Acme"


def test_jobSummary_titleOnly_returnsTitle() -> None:
    job = JobDescription(raw_text="x", title="Senior Engineer")
    assert _job_summary(job) == "Senior Engineer"


def test_jobSummary_companyOnly_returnsCompany() -> None:
    job = JobDescription(raw_text="x", company="Acme")
    assert _job_summary(job) == "Acme"


def test_jobSummary_neither_fallsBackToTruncatedRawText() -> None:
    job = JobDescription(raw_text="Looking for someone great, no title or company extracted")
    assert _job_summary(job) == job.raw_text[:120]


def test_jobSummary_neitherAndBlankRawText_returnsNone() -> None:
    job = JobDescription(raw_text="   ")
    assert _job_summary(job) is None
