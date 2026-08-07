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
    JobDescription,
    ModelUsage,
    ResumeDraft,
    ResumeSection,
    RunUsage,
    TrustReport,
    VerifiedClaim,
)
from trustresume.storage import (
    ChunkRepository,
    DocumentRepository,
    EvaluationRepository,
    JobDocumentRepository,
    JobRepository,
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


# --- DocumentRepository: filename identity + job-scoped eligibility --------


def test_documentRepository_findByFilename_scopedByUser(db: sqlite3.Connection) -> None:
    users = UserRepository(db)
    users.create("A", user_id="u1")
    users.create("B", user_id="u2")
    docs = DocumentRepository(db)
    doc_id = docs.create(
        user_id="u1", filename="r.txt", document_type=DocumentType.RESUME, content_hash="h1"
    )

    found = docs.find_by_filename(user_id="u1", filename="r.txt")
    assert found is not None and found["id"] == doc_id
    assert docs.find_by_filename(user_id="u1", filename="missing.txt") is None
    # Same filename, different (or no) document for another user.
    assert docs.find_by_filename(user_id="u2", filename="r.txt") is None


def test_documentRepository_updateContentHash_preservesIdChangesHashAndFilename(
    db: sqlite3.Connection,
) -> None:
    users = UserRepository(db)
    users.create("A", user_id="u1")
    docs = DocumentRepository(db)
    doc_id = docs.create(
        user_id="u1", filename="r.txt", document_type=DocumentType.RESUME, content_hash="h1"
    )

    docs.update_content_hash(user_id="u1", document_id=doc_id, content_hash="h2", filename="r.txt")

    row = docs.find_by_filename(user_id="u1", filename="r.txt")
    assert row is not None
    assert row["id"] == doc_id  # same logical document, not a new row
    assert row["content_hash"] == "h2"


def test_documentRepository_listEligibleDocumentIds_genericPoolUnionedWithJobLinked(
    db: sqlite3.Connection,
) -> None:
    users = UserRepository(db)
    users.create("A", user_id="u1")
    docs = DocumentRepository(db)
    generic_id = docs.create(
        user_id="u1", filename="generic.txt", document_type=DocumentType.RESUME, content_hash="h1"
    )
    linked_id = docs.create(
        user_id="u1", filename="linked.txt", document_type=DocumentType.RESUME, content_hash="h2"
    )
    other_job_only_id = docs.create(
        user_id="u1", filename="other.txt", document_type=DocumentType.RESUME, content_hash="h3"
    )
    jobs = JobRepository(db)
    job_id = jobs.create(
        user_id="u1", raw_posting="x", job=JobDescription(raw_text="x"), summary=None
    )
    other_job_id = jobs.create(
        user_id="u1", raw_posting="y", job=JobDescription(raw_text="y"), summary=None
    )
    job_documents = JobDocumentRepository(db)
    job_documents.link(job_id=job_id, document_id=linked_id)
    job_documents.link(job_id=other_job_id, document_id=other_job_only_id)

    eligible_for_job = set(docs.list_eligible_document_ids(user_id="u1", job_id=job_id))
    eligible_generic_only = set(docs.list_eligible_document_ids(user_id="u1", job_id=None))

    # Generic pool (unlinked to ANY job) + this job's own link — not the
    # other job's link, since "generic" means unlinked to any job, not just
    # unlinked to this one.
    assert eligible_for_job == {generic_id, linked_id}
    assert eligible_generic_only == {generic_id}


# --- JobRepository -----------------------------------------------------------


def test_jobRepository_createGetUpdateDelete(db: sqlite3.Connection) -> None:
    users = UserRepository(db)
    users.create("A", user_id="u1")
    jobs = JobRepository(db)
    job = JobDescription(raw_text="Need a Python engineer", title="Engineer", company="Acme")
    job_id = jobs.create(
        user_id="u1", raw_posting=job.raw_text, job=job, summary="Engineer at Acme"
    )

    row = jobs.get(user_id="u1", job_id=job_id)
    assert row is not None
    assert row["title"] == "Engineer"
    assert row["company"] == "Acme"
    assert row["summary"] == "Engineer at Acme"
    assert JobDescription.model_validate_json(row["job_description_json"]) == job

    assert len(jobs.list_for_user("u1")) == 1
    assert jobs.exists(user_id="u1", job_id=job_id) is True
    assert jobs.exists(user_id="u1", job_id="missing") is False

    updated_job = JobDescription(raw_text="Need a Staff engineer", title="Staff Engineer")
    assert (
        jobs.update(
            user_id="u1",
            job_id=job_id,
            raw_posting=updated_job.raw_text,
            job=updated_job,
            summary="Staff Engineer",
        )
        is True
    )
    updated_row = jobs.get(user_id="u1", job_id=job_id)
    assert updated_row is not None
    assert updated_row["title"] == "Staff Engineer"
    assert (
        jobs.update(
            user_id="u1",
            job_id="missing",
            raw_posting="x",
            job=JobDescription(raw_text="x"),
            summary=None,
        )
        is False
    )

    jobs.delete(user_id="u1", job_id=job_id)
    assert jobs.get(user_id="u1", job_id=job_id) is None


def test_jobRepository_isolatedByUser(db: sqlite3.Connection) -> None:
    users = UserRepository(db)
    users.create("A", user_id="u1")
    users.create("B", user_id="u2")
    jobs = JobRepository(db)
    job_id = jobs.create(
        user_id="u1", raw_posting="x", job=JobDescription(raw_text="x"), summary=None
    )

    assert jobs.get(user_id="u2", job_id=job_id) is None
    assert jobs.list_for_user("u2") == []
    assert jobs.exists(user_id="u2", job_id=job_id) is False


# --- JobDocumentRepository ---------------------------------------------------


def test_jobDocumentRepository_linkIsIdempotentAndUnlinkRemovesOnlyTheLink(
    db: sqlite3.Connection,
) -> None:
    users = UserRepository(db)
    users.create("A", user_id="u1")
    docs = DocumentRepository(db)
    doc_id = docs.create(
        user_id="u1", filename="r.txt", document_type=DocumentType.RESUME, content_hash="h1"
    )
    jobs = JobRepository(db)
    job_id = jobs.create(
        user_id="u1", raw_posting="x", job=JobDescription(raw_text="x"), summary=None
    )
    job_documents = JobDocumentRepository(db)

    job_documents.link(job_id=job_id, document_id=doc_id)
    job_documents.link(job_id=job_id, document_id=doc_id)  # idempotent, no error

    assert job_documents.list_document_ids_for_job(job_id) == [doc_id]
    assert job_documents.list_job_ids_for_document(doc_id) == [job_id]

    job_documents.unlink(job_id=job_id, document_id=doc_id)

    assert job_documents.list_document_ids_for_job(job_id) == []
    # The document itself is untouched by unlinking.
    assert docs.find_by_filename(user_id="u1", filename="r.txt") is not None


def test_jobDocumentRepository_cascadesOnJobDeleteAndOnDocumentDelete(
    db: sqlite3.Connection,
) -> None:
    users = UserRepository(db)
    users.create("A", user_id="u1")
    docs = DocumentRepository(db)
    doc_id = docs.create(
        user_id="u1", filename="r.txt", document_type=DocumentType.RESUME, content_hash="h1"
    )
    jobs = JobRepository(db)
    job_id = jobs.create(
        user_id="u1", raw_posting="x", job=JobDescription(raw_text="x"), summary=None
    )
    job_documents = JobDocumentRepository(db)
    job_documents.link(job_id=job_id, document_id=doc_id)

    jobs.delete(user_id="u1", job_id=job_id)
    assert job_documents.list_document_ids_for_job(job_id) == []

    # Re-link a fresh job, then delete the document instead.
    job_id_2 = jobs.create(
        user_id="u1", raw_posting="y", job=JobDescription(raw_text="y"), summary=None
    )
    job_documents.link(job_id=job_id_2, document_id=doc_id)
    docs.delete(user_id="u1", document_id=doc_id)
    assert job_documents.list_job_ids_for_document(doc_id) == []


# --- ChunkRepository.search_keywords: document_ids filter -------------------


def test_chunkRepository_searchKeywords_documentIdsFilter(db: sqlite3.Connection) -> None:
    users = UserRepository(db)
    users.create("A", user_id="u1")
    docs = DocumentRepository(db)
    doc1 = docs.create(
        user_id="u1", filename="a.txt", document_type=DocumentType.RESUME, content_hash="h1"
    )
    doc2 = docs.create(
        user_id="u1", filename="b.txt", document_type=DocumentType.RESUME, content_hash="h2"
    )
    chunks = ChunkRepository(db)
    chunks.add(
        EvidenceChunk(chunk_id="c1", user_id="u1", document_id=doc1, text="Kubernetes expert"),
        chunk_index=0,
    )
    chunks.add(
        EvidenceChunk(chunk_id="c2", user_id="u1", document_id=doc2, text="Kubernetes expert too"),
        chunk_index=0,
    )

    all_hits = chunks.search_keywords(user_id="u1", query="Kubernetes", limit=10)
    scoped_hits = chunks.search_keywords(
        user_id="u1", query="Kubernetes", limit=10, document_ids=[doc1]
    )
    empty_hits = chunks.search_keywords(user_id="u1", query="Kubernetes", limit=10, document_ids=[])

    assert len(all_hits) == 2
    assert [h["chunk_id"] for h in scoped_hits] == ["c1"]
    assert empty_hits == []


# --- ResumeRepository: export bytes, job linkage, rejection data -----------


def test_resumeRepository_create_persistsJobIdExportsAndRejectionData(
    db: sqlite3.Connection,
) -> None:
    users = UserRepository(db)
    users.create("A", user_id="u1")
    jobs = JobRepository(db)
    job_id = jobs.create(
        user_id="u1", raw_posting="x", job=JobDescription(raw_text="x"), summary=None
    )
    resumes = ResumeRepository(db)
    draft = ResumeDraft(summary="s", sections=[ResumeSection(heading="Skills", bullets=["Python"])])

    resume_id = resumes.create(
        user_id="u1",
        draft=draft,
        job_title="Engineer",
        trust_score=62.0,
        ats_score=78.0,
        passed=False,
        job_id=job_id,
        pdf_bytes=b"%PDF-1.4 fake",
        markdown_text="# s",
        rejection_reason="Trust score 62 (needs >= 90).",
        improvement_suggestions="Improve evidence-grounding.",
    )

    row = resumes.get_row(user_id="u1", resume_id=resume_id)
    assert row is not None
    assert row["job_id"] == job_id
    assert bytes(row["pdf_bytes"]) == b"%PDF-1.4 fake"
    assert row["markdown_text"] == "# s"
    assert row["rejection_reason"] == "Trust score 62 (needs >= 90)."
    assert row["improvement_suggestions"] == "Improve evidence-grounding."
    # Isolation still holds for the new row-returning accessor.
    assert resumes.get_row(user_id="u2", resume_id=resume_id) is None


def test_resumeRepository_create_persistsUsageColumns(db: sqlite3.Connection) -> None:
    users = UserRepository(db)
    users.create("A", user_id="u1")
    resumes = ResumeRepository(db)
    usage = RunUsage(
        models=[
            ModelUsage(model="m1", input_tokens=100, output_tokens=40, calls=3),
            ModelUsage(model="m2", input_tokens=10, output_tokens=5, calls=1),
        ],
        total_duration_ms=1234.5,
        cost_usd=0.0031,
    )

    resume_id = resumes.create(
        user_id="u1",
        draft=ResumeDraft(summary="s"),
        job_title=None,
        trust_score=90.0,
        ats_score=90.0,
        passed=True,
        usage=usage,
    )

    row = resumes.get_row(user_id="u1", resume_id=resume_id)
    assert row is not None
    assert row["input_tokens"] == 110  # summed across both models
    assert row["output_tokens"] == 45
    assert row["llm_calls"] == 4
    assert row["cost_usd"] == 0.0031
    assert row["duration_ms"] == 1234.5


def test_resumeRepository_create_withoutUsage_leavesColumnsNullNotZero(
    db: sqlite3.Connection,
) -> None:
    """Unmeasured must stay distinguishable from measured-and-free."""
    users = UserRepository(db)
    users.create("A", user_id="u1")
    resumes = ResumeRepository(db)

    resume_id = resumes.create(
        user_id="u1",
        draft=ResumeDraft(summary="s"),
        job_title=None,
        trust_score=90.0,
        ats_score=90.0,
        passed=True,
    )

    row = resumes.get_row(user_id="u1", resume_id=resume_id)
    assert row is not None
    assert row["llm_calls"] is None
    assert row["cost_usd"] is None
    assert row["duration_ms"] is None


def test_resumeRepository_listForJob_scopedToJobAndUser(db: sqlite3.Connection) -> None:
    users = UserRepository(db)
    users.create("A", user_id="u1")
    jobs = JobRepository(db)
    job_id = jobs.create(
        user_id="u1", raw_posting="x", job=JobDescription(raw_text="x"), summary=None
    )
    other_job_id = jobs.create(
        user_id="u1", raw_posting="y", job=JobDescription(raw_text="y"), summary=None
    )
    resumes = ResumeRepository(db)
    resumes.create(
        user_id="u1",
        draft=ResumeDraft(iteration=0),
        job_title=None,
        trust_score=90.0,
        ats_score=90.0,
        passed=True,
        job_id=job_id,
    )
    resumes.create(
        user_id="u1",
        draft=ResumeDraft(iteration=0),
        job_title=None,
        trust_score=90.0,
        ats_score=90.0,
        passed=True,
        job_id=other_job_id,
    )

    for_job = resumes.list_for_job(user_id="u1", job_id=job_id)
    assert len(for_job) == 1
    assert resumes.list_for_job(user_id="u2", job_id=job_id) == []


def test_resumeRepository_jobDeletedSetsJobIdNull_keepsJobTitleSnapshot(
    db: sqlite3.Connection,
) -> None:
    users = UserRepository(db)
    users.create("A", user_id="u1")
    jobs = JobRepository(db)
    job_id = jobs.create(
        user_id="u1", raw_posting="x", job=JobDescription(raw_text="x"), summary=None
    )
    resumes = ResumeRepository(db)
    resume_id = resumes.create(
        user_id="u1",
        draft=ResumeDraft(iteration=0),
        job_title="Engineer",
        trust_score=90.0,
        ats_score=90.0,
        passed=True,
        job_id=job_id,
    )

    jobs.delete(user_id="u1", job_id=job_id)

    row = resumes.get_row(user_id="u1", resume_id=resume_id)
    assert row is not None
    assert row["job_id"] is None
    assert row["job_title"] == "Engineer"
