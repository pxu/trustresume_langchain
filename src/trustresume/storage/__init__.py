"""Structured storage layer (SQLite): users, documents, generated resumes, evaluations.

The SQLite half of the hybrid store (ADR-0001). ``schema`` owns the DDL and
connection factory; ``repositories`` exposes one user-scoped repository per
aggregate. The canonical schema doc is the *original* repo's
``architecture/data-model/README.md`` — that document wasn't ported here (see
SYNC.md); ``schema.py`` is the executable source of truth in this repo.

Milestone M2 (storage + retrieval).
"""

from __future__ import annotations

from .repositories import (
    CandidateProfileCacheEntry,
    CandidateProfileRepository,
    ChunkRepository,
    DocumentRepository,
    EvaluationRepository,
    JobDocumentRepository,
    JobRepository,
    ResumeRepository,
    UserRepository,
)
from .schema import connect, init_db

__all__ = [
    "connect",
    "init_db",
    "UserRepository",
    "DocumentRepository",
    "ChunkRepository",
    "CandidateProfileRepository",
    "CandidateProfileCacheEntry",
    "ResumeRepository",
    "EvaluationRepository",
    "JobRepository",
    "JobDocumentRepository",
]
