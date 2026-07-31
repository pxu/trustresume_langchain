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
    docs.create(
        user_id="u1", filename="a.pdf", document_type=DocumentType.RESUME, content_hash="h1"
    )
    docs.create(
        user_id="u2", filename="b.pdf", document_type=DocumentType.RESUME, content_hash="h1"
    )

    u1_docs = docs.list_for_user("u1")
    assert len(u1_docs) == 1
    assert u1_docs[0]["filename"] == "a.pdf"


def test_chunkRepository_addListAndDeleteReturnsIds(db: sqlite3.Connection) -> None:
    users = UserRepository(db)
    users.create("A", user_id="u1")
    docs = DocumentRepository(db)
    doc_id = docs.create(
        user_id="u1", filename="a.pdf", document_type=DocumentType.RESUME, content_hash="h1"
    )

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
    doc_id = docs.create(
        user_id="u1", filename="a.pdf", document_type=DocumentType.RESUME, content_hash="h1"
    )
    chunks = ChunkRepository(db)
    chunks.add(
        EvidenceChunk(chunk_id="c0", user_id="u1", document_id=doc_id, text="secret"),
        chunk_index=0,
    )

    assert chunks.list_for_user("u2") == []


def test_chunkRepository_searchKeywords_matchesExactTerm(db: sqlite3.Connection) -> None:
    users = UserRepository(db)
    users.create("A", user_id="u1")
    docs = DocumentRepository(db)
    doc_id = docs.create(
        user_id="u1", filename="r.txt", document_type=DocumentType.RESUME, content_hash="h1"
    )
    chunks = ChunkRepository(db)
    chunks.add(
        EvidenceChunk(
            chunk_id="c0", user_id="u1", document_id=doc_id, text="Built services on AWS Lambda"
        ),
        chunk_index=0,
    )
    chunks.add(
        EvidenceChunk(chunk_id="c1", user_id="u1", document_id=doc_id, text="Led a team"),
        chunk_index=1,
    )

    hits = chunks.search_keywords(user_id="u1", query="Lambda", limit=5)
    assert [h["chunk_id"] for h in hits] == ["c0"]


def test_chunkRepository_searchKeywords_sanitizesPunctuation(db: sqlite3.Connection) -> None:
    """FTS5's MATCH syntax breaks on raw punctuation (a real job-description
    query might contain "AI/ML" or parentheses) — search_keywords must not
    raise, and should still match the alphanumeric tokens it can extract.
    """
    users = UserRepository(db)
    users.create("A", user_id="u1")
    docs = DocumentRepository(db)
    doc_id = docs.create(
        user_id="u1", filename="r.txt", document_type=DocumentType.RESUME, content_hash="h1"
    )
    chunks = ChunkRepository(db)
    chunks.add(
        EvidenceChunk(chunk_id="c0", user_id="u1", document_id=doc_id, text="Python and AWS work"),
        chunk_index=0,
    )

    hits = chunks.search_keywords(user_id="u1", query="AI/ML (Python) engineer!", limit=5)
    assert [h["chunk_id"] for h in hits] == ["c0"]


def test_chunkRepository_searchKeywords_noAlphanumericTokens_returnsEmpty(
    db: sqlite3.Connection,
) -> None:
    users = UserRepository(db)
    users.create("A", user_id="u1")
    chunks = ChunkRepository(db)
    assert chunks.search_keywords(user_id="u1", query="@#$%", limit=5) == []


def test_chunkRepository_searchKeywords_isolatesUsers(db: sqlite3.Connection) -> None:
    users = UserRepository(db)
    users.create("A", user_id="u1")
    users.create("B", user_id="u2")
    docs = DocumentRepository(db)
    doc_id = docs.create(
        user_id="u1", filename="r.txt", document_type=DocumentType.RESUME, content_hash="h1"
    )
    chunks = ChunkRepository(db)
    chunks.add(
        EvidenceChunk(chunk_id="c0", user_id="u1", document_id=doc_id, text="secret Kubernetes"),
        chunk_index=0,
    )

    assert chunks.search_keywords(user_id="u2", query="Kubernetes", limit=5) == []


def test_chunkRepository_searchKeywords_updateSyncsIndex(db: sqlite3.Connection) -> None:
    """The chunks_fts external-content triggers must keep the FTS index in
    sync on UPDATE too, not just insert/delete — otherwise editing a chunk's
    text in place would desync the keyword index from what's actually there.
    """
    users = UserRepository(db)
    users.create("A", user_id="u1")
    docs = DocumentRepository(db)
    doc_id = docs.create(
        user_id="u1", filename="r.txt", document_type=DocumentType.RESUME, content_hash="h1"
    )
    chunks = ChunkRepository(db)
    chunks.add(
        EvidenceChunk(chunk_id="c0", user_id="u1", document_id=doc_id, text="Kubernetes expert"),
        chunk_index=0,
    )
    assert chunks.search_keywords(user_id="u1", query="Kubernetes", limit=5)

    db.execute("UPDATE chunks SET text = 'Terraform expert' WHERE chunk_id = 'c0'")
    db.commit()

    assert chunks.search_keywords(user_id="u1", query="Kubernetes", limit=5) == []
    hits = chunks.search_keywords(user_id="u1", query="Terraform", limit=5)
    assert [h["chunk_id"] for h in hits] == ["c0"]


def test_chunkRepository_searchKeywords_tiedRank_breaksTiesByChunkId(
    db: sqlite3.Connection,
) -> None:
    """Two chunks with identical BM25 rank (here: identical text) must sort
    deterministically rather than in unspecified row order — HybridRetriever
    treats keyword-list position as ground truth for RRF fusion, so an
    unstable tie order would make fusion results non-reproducible.
    """
    users = UserRepository(db)
    users.create("A", user_id="u1")
    docs = DocumentRepository(db)
    doc_id = docs.create(
        user_id="u1", filename="r.txt", document_type=DocumentType.RESUME, content_hash="h1"
    )
    chunks = ChunkRepository(db)
    chunks.add(
        EvidenceChunk(chunk_id="c9", user_id="u1", document_id=doc_id, text="Kubernetes expert"),
        chunk_index=0,
    )
    chunks.add(
        EvidenceChunk(chunk_id="c1", user_id="u1", document_id=doc_id, text="Kubernetes expert"),
        chunk_index=1,
    )

    hits = chunks.search_keywords(user_id="u1", query="Kubernetes", limit=5)

    assert [h["chunk_id"] for h in hits] == ["c1", "c9"]


def test_chunkRepository_searchKeywords_deleteSyncsIndex(db: sqlite3.Connection) -> None:
    """The chunks_fts external-content triggers must keep the FTS index in
    sync on delete, not just insert — otherwise a deleted chunk keeps
    matching keyword searches.
    """
    users = UserRepository(db)
    users.create("A", user_id="u1")
    docs = DocumentRepository(db)
    doc_id = docs.create(
        user_id="u1", filename="r.txt", document_type=DocumentType.RESUME, content_hash="h1"
    )
    chunks = ChunkRepository(db)
    chunks.add(
        EvidenceChunk(chunk_id="c0", user_id="u1", document_id=doc_id, text="Kubernetes expert"),
        chunk_index=0,
    )
    assert chunks.search_keywords(user_id="u1", query="Kubernetes", limit=5)

    chunks.delete_for_document(user_id="u1", document_id=doc_id)

    assert chunks.search_keywords(user_id="u1", query="Kubernetes", limit=5) == []


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
    doc_id = docs.create(
        user_id="u1", filename="a.pdf", document_type=DocumentType.RESUME, content_hash="h1"
    )
    chunks = ChunkRepository(db)
    chunks.add(
        EvidenceChunk(chunk_id="c0", user_id="u1", document_id=doc_id, text="x"),
        chunk_index=0,
    )

    # Deleting the user cascades to documents and chunks (PRAGMA foreign_keys=ON).
    db.execute("DELETE FROM users WHERE id = ?", ("u1",))
    db.commit()
    assert chunks.list_for_user("u1") == []
