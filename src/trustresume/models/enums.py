"""Enumerations shared across TrustResume schemas.

`StrEnum` (Python 3.11+) gives readable, JSON-friendly values — a serialized
model shows ``"SUPPORTED"`` rather than an opaque integer, which matters for
the explainable Trust Report that is the project's research contribution
(ADR-0004).

Milestone M1 (shared models) — the data contracts every other package imports.
"""

from __future__ import annotations

from enum import StrEnum


class ClaimStatus(StrEnum):
    """How well a single resume claim is backed by retrieved evidence."""

    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"


class ClaimCategory(StrEnum):
    """The kind of assertion a claim makes.

    Unsupported skills, experience, certifications, and achievements are the
    four categories the Trust Harness flags as hallucinations (ADR-0004).
    """

    SKILL = "SKILL"
    EXPERIENCE = "EXPERIENCE"
    CERTIFICATION = "CERTIFICATION"
    ACHIEVEMENT = "ACHIEVEMENT"
    OTHER = "OTHER"


class DocumentType(StrEnum):
    """Type of candidate source document an evidence chunk came from.

    Mirrors the ``document_type`` metadata field on every Chroma chunk
    (ADR-0001).
    """

    RESUME = "RESUME"
    PROJECT_REPORT = "PROJECT_REPORT"
    STAR_STORY = "STAR_STORY"
    CERTIFICATION = "CERTIFICATION"
    COVER_LETTER = "COVER_LETTER"
    OTHER = "OTHER"


class SeniorityLevel(StrEnum):
    """Coarse seniority the Job Description agent parses from a job posting."""

    INTERN = "INTERN"
    JUNIOR = "JUNIOR"
    MID = "MID"
    SENIOR = "SENIOR"
    LEAD = "LEAD"
    MANAGER = "MANAGER"
    UNKNOWN = "UNKNOWN"
