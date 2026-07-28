"""Unit tests for the SQLite repositories.

Focus areas: round-tripping models through the DB, and the ADR-0001 isolation
guarantee that every read is scoped by ``user_id``.
"""

from __future__ import annotations

import sqlite3

from trustresume.models import (
    ATSReport,
    ClaimStatus,
    DocumentType,
    EvidenceChunk,
    ResumeDraft,
    ResumeSection,
    TrustReport,
    VerifiedClaim,
)
from trustresume.storage import (
    ChunkRepository,
    DocumentRepository,
    EvaluationRepository,
    ResumeRepository,
    UserRepository,
)


def test_userRepository_createAndExists(db: sqlite3.Connection) -> None:
    users = UserRepository(db)
    uid = users.create("Ada Lovelace", user_id="u1")

    assert uid == "u1"
    assert users.exists("u1") is True
    assert users.exists("missing") is False


def test_documentRepository_listScopedByUser(db: sqlite3.Connection) -> None:
    users = UserRepository(db)
    users.create("A", user_id="u1")
    users.create("B", user_id="u2")
    docs = DocumentRepository(db)
    docs.create(user_id="u1", filename="a.pdf", document_type=DocumentType.RESUME)
    docs.create(user_id="u2", filename="b.pdf", document_type=DocumentType.RESUME)

    u1_docs = docs.list_for_user("u1")
    assert len(u1_docs) == 1
    assert u1_docs[0]["filename"] == "a.pdf"


def test_chunkRepository_addListAndDeleteReturnsIds(db: sqlite3.Connection) -> None:
    users = UserRepository(db)
    users.create("A", user_id="u1")
    docs = DocumentRepository(db)
    doc_id = docs.create(user_id="u1", filename="a.pdf", document_type=DocumentType.RESUME)

    chunks = ChunkRepository(db)
    for i in range(3):
        chunks.add(
            EvidenceChunk(
                chunk_id=f"c{i}",
                user_id="u1",
                document_id=doc_id,
                document_type=DocumentType.RESUME,
                source_document="a.pdf",
                text=f"chunk {i}",
            ),
            chunk_index=i,
        )

    assert len(chunks.list_for_user("u1")) == 3
    deleted = chunks.delete_for_document(user_id="u1", document_id=doc_id)
    assert sorted(deleted) == ["c0", "c1", "c2"]
    assert chunks.list_for_user("u1") == []


def test_chunkRepository_isolation_otherUserSeesNothing(db: sqlite3.Connection) -> None:
    users = UserRepository(db)
    users.create("A", user_id="u1")
    users.create("B", user_id="u2")
    docs = DocumentRepository(db)
    doc_id = docs.create(user_id="u1", filename="a.pdf", document_type=DocumentType.RESUME)
    chunks = ChunkRepository(db)
    chunks.add(
        EvidenceChunk(chunk_id="c0", user_id="u1", document_id=doc_id, text="secret"),
        chunk_index=0,
    )

    assert chunks.list_for_user("u2") == []


def test_resumeRepository_roundTrip(db: sqlite3.Connection) -> None:
    users = UserRepository(db)
    users.create("A", user_id="u1")
    resumes = ResumeRepository(db)
    draft = ResumeDraft(
        summary="Backend engineer.",
        sections=[ResumeSection(heading="Skills", bullets=["Python"])],
        iteration=2,
    )
    resume_id = resumes.create(
        user_id="u1",
        draft=draft,
        job_title="Backend Engineer",
        trust_score=91.0,
        ats_score=87.0,
        passed=True,
    )

    loaded = resumes.get(user_id="u1", resume_id=resume_id)
    assert loaded == draft
    # Isolation: another user can't fetch it.
    assert resumes.get(user_id="u2", resume_id=resume_id) is None


def test_evaluationRepository_roundTrip(db: sqlite3.Connection) -> None:
    users = UserRepository(db)
    users.create("A", user_id="u1")
    resumes = ResumeRepository(db)
    resume_id = resumes.create(
        user_id="u1",
        draft=ResumeDraft(iteration=0),
        job_title=None,
        trust_score=50.0,
        ats_score=50.0,
        passed=False,
    )
    evals = EvaluationRepository(db)
    trust = TrustReport.from_claims(
        [VerifiedClaim(text="Knows Python", status=ClaimStatus.SUPPORTED)],
        iteration=0,
    )
    ats = ATSReport(score=80.0, matched_keywords=["python"])
    evals.create(user_id="u1", resume_id=resume_id, trust=trust, ats=ats)

    pairs = evals.list_for_resume(user_id="u1", resume_id=resume_id)
    assert len(pairs) == 1
    assert pairs[0]["trust"].score == trust.score
    assert pairs[0]["ats"].matched_keywords == ["python"]


def test_foreignKey_cascadeDeletesChunks(db: sqlite3.Connection) -> None:
    users = UserRepository(db)
    users.create("A", user_id="u1")
    docs = DocumentRepository(db)
    doc_id = docs.create(user_id="u1", filename="a.pdf", document_type=DocumentType.RESUME)
    chunks = ChunkRepository(db)
    chunks.add(
        EvidenceChunk(chunk_id="c0", user_id="u1", document_id=doc_id, text="x"),
        chunk_index=0,
    )

    # Deleting the user cascades to documents and chunks (PRAGMA foreign_keys=ON).
    db.execute("DELETE FROM users WHERE id = ?", ("u1",))
    db.commit()
    assert chunks.list_for_user("u1") == []
