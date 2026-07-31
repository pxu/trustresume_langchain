"""End-to-end integration test for TrustResumeApp (M7).

Exercises the whole stack wired together — ingestion → retrieval → the agents
→ orchestrator → persistence — with in-memory stores, the deterministic fake
embedder, and a scripted chat model driving the LLM agents. No network, no
model download, no AWS.

Unlike pydantic-ai's ``TestModel()``, which auto-synthesizes valid structured
output for whatever schema an agent requests, LangChain's fakes only replay a
pre-scripted message sequence — so each test scripts exactly the LLM calls its
code path will make, in order, with content engineered to pass the default
quality gate (Trust >= 90, ATS >= 85) on the first attempt. That keeps the
call count deterministic and low (one ``generate()`` with a fresh candidate
profile is 4 calls: Job Description, Candidate Profile, Resume Writer, Trust
Harness — ATS Evaluation is deterministic, no LLM call).
"""

from __future__ import annotations

import chromadb
import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolCall

from tests.fakes import FakeEmbeddings, FakeToolCallingChatModel
from trustresume.api.app_service import TrustResumeApp, build_default_app
from trustresume.models import DocumentType
from trustresume.storage import connect

# Scripted so the pipeline passes the default gate on the first attempt:
# job.keywords + the draft's content overlap fully (ATS = 100), and the one
# claim is SUPPORTED (Trust = 100).
_JOB_DESCRIPTION_CALL = AIMessage(
    content="",
    tool_calls=[
        ToolCall(
            name="JobDescription",
            args={
                "raw_text": "ignored — overwritten by the agent",
                "title": "Senior Python Engineer",
                "keywords": ["python", "aws"],
            },
            id="call_jd",
        )
    ],
)
_CANDIDATE_PROFILE_CALL = AIMessage(
    content="",
    tool_calls=[
        ToolCall(
            name="CandidateProfile",
            args={"name": "Ada", "skills": ["python", "aws"]},
            id="call_cp",
        )
    ],
)
_RESUME_DRAFT_CALL = AIMessage(
    content="",
    tool_calls=[
        ToolCall(
            name="_DraftExtraction",
            args={
                "summary": "Experienced python and aws engineer.",
                "sections": [{"heading": "Skills", "bullets": ["python", "aws"]}],
            },
            id="call_resume",
        )
    ],
)
_TRUST_CALL = AIMessage(
    content="",
    tool_calls=[
        ToolCall(
            name="_ClaimExtraction",
            args={
                "claims": [
                    {
                        "text": "Built python services on AWS",
                        "category": "SKILL",
                        "status": "SUPPORTED",
                    }
                ]
            },
            id="call_trust",
        )
    ],
)

# One full `generate()` call with a fresh (not-yet-cached) candidate profile.
_FULL_GENERATION = [_JOB_DESCRIPTION_CALL, _CANDIDATE_PROFILE_CALL, _RESUME_DRAFT_CALL, _TRUST_CALL]
# A `generate()` call that hits a warm candidate-profile cache (no CP call).
_CACHED_PROFILE_GENERATION = [_JOB_DESCRIPTION_CALL, _RESUME_DRAFT_CALL, _TRUST_CALL]


def _app_with_scripted_calls(*call_groups: list[AIMessage]) -> TrustResumeApp:
    messages = [msg for group in call_groups for msg in group]
    model = FakeToolCallingChatModel(messages=iter(messages))
    return TrustResumeApp(
        connection=connect(":memory:"),
        chroma_client=chromadb.EphemeralClient(),
        embedder=FakeEmbeddings(),
        model=model,
    )


@pytest.fixture
def app() -> TrustResumeApp:
    """An in-memory app scripted for exactly one full `generate()` call."""
    return _app_with_scripted_calls(_FULL_GENERATION)


def test_buildDefaultApp_unknownProvider_raises() -> None:
    from trustresume.api.model_factory import LLMConfig

    with pytest.raises(ValueError, match="unknown LLM provider"):
        build_default_app(llm_config=LLMConfig(provider="nope"))


def test_close_releasesConnectionAndChromaClient() -> None:
    """``close()`` must actually release both underlying resources, not just
    exist — a closed sqlite3.Connection raises ProgrammingError on any further
    use, which is exactly the signal we want to confirm it's really closed.
    """
    import sqlite3

    app_facade = _app_with_scripted_calls([])

    app_facade.close()

    with pytest.raises(sqlite3.ProgrammingError):
        app_facade._connection.execute("SELECT 1")


def test_contextManager_closesOnExit() -> None:
    import sqlite3

    with _app_with_scripted_calls([]) as app_facade:
        app_facade.ensure_user("Ada", user_id="u1")

    with pytest.raises(sqlite3.ProgrammingError):
        app_facade._connection.execute("SELECT 1")


def test_ensureUser_isIdempotent(app: TrustResumeApp) -> None:
    assert app.ensure_user("Ada", user_id="u1") == "u1"
    # Second call must not raise (user already exists).
    assert app.ensure_user("Ada", user_id="u1") == "u1"


def test_ensureUser_concurrentRace_doesNotRaise(app: TrustResumeApp) -> None:
    # The exists()-then-create() check isn't one atomic transaction, so two
    # concurrent first-calls for the same new user_id can both see
    # exists() == False and both try to create it; the second must lose to
    # the primary-key constraint quietly rather than surfacing an
    # sqlite3.IntegrityError, the same race DocumentRepository already
    # handles for content-hash collisions. Simulated here by pre-creating the
    # row directly, then forcing ensure_user's own exists() check to lie
    # (as if it ran before the racing insert landed).
    app._users.create("Ada", user_id="u1")
    app._users.exists = lambda user_id: False  # type: ignore[method-assign]

    assert app.ensure_user("Ada", user_id="u1") == "u1"


def test_addDocument_thenListed(app: TrustResumeApp) -> None:
    app.ensure_user("Ada", user_id="u1")
    app.add_document(
        user_id="u1",
        filename="resume.txt",
        text="Built Python services on AWS. Led a team of five.",
        document_type=DocumentType.RESUME,
    )
    docs = app.list_documents("u1")
    assert len(docs) == 1
    assert docs[0]["filename"] == "resume.txt"


def test_addDocumentBytes_thenListed(app: TrustResumeApp) -> None:
    app.ensure_user("Ada", user_id="u1")
    app.add_document_bytes(
        user_id="u1",
        filename="resume.txt",
        data=b"Built Python services on AWS.",
        document_type=DocumentType.RESUME,
    )
    docs = app.list_documents("u1")
    assert len(docs) == 1
    assert docs[0]["filename"] == "resume.txt"


def test_deleteDocument_removesItAndReturnsTrue(app: TrustResumeApp) -> None:
    app.ensure_user("Ada", user_id="u1")
    doc_id = app.add_document(user_id="u1", filename="resume.txt", text="Built Python on AWS.")

    assert app.delete_document(user_id="u1", document_id=doc_id) is True
    assert app.list_documents("u1") == []


def test_deleteDocument_unknownId_returnsFalseWithoutRaising(app: TrustResumeApp) -> None:
    app.ensure_user("Ada", user_id="u1")
    assert app.delete_document(user_id="u1", document_id="nope") is False


def test_deleteDocument_wrongUser_returnsFalseAndLeavesDocumentIntact(
    app: TrustResumeApp,
) -> None:
    app.ensure_user("Ada", user_id="u1")
    app.ensure_user("Bob", user_id="u2")
    doc_id = app.add_document(user_id="u1", filename="resume.txt", text="Built Python on AWS.")

    assert app.delete_document(user_id="u2", document_id=doc_id) is False
    assert len(app.list_documents("u1")) == 1


def test_searchEvidence_returnsUserScopedChunks(app: TrustResumeApp) -> None:
    # Unique user ids (not "u1"/"u2") — chromadb.EphemeralClient() caches its
    # underlying storage by collection name across instances within a
    # process (see CLAUDE.md's testing-model note), so reusing ids other
    # tests in this module already write under would leak chunks across
    # "fresh" per-test apps.
    app.ensure_user("Ada", user_id="search-u1")
    app.ensure_user("Bob", user_id="search-u2")
    app.add_document(user_id="search-u1", filename="r.txt", text="Built Python services on AWS.")
    app.add_document(user_id="search-u2", filename="r.txt", text="Cooked five-star meals.")

    result = app.search_evidence(user_id="search-u1", query="Python AWS", limit=5)
    assert result.user_id == "search-u1"
    assert len(result.chunks) == 1
    assert "Python" in result.chunks[0].text


def test_generate_runsPipelineAndPersists(app: TrustResumeApp) -> None:
    app.ensure_user("Ada", user_id="u1")
    app.add_document(
        user_id="u1",
        filename="resume.txt",
        text="Built Python services on AWS. Led a team of five engineers.",
        document_type=DocumentType.RESUME,
    )

    state = app.generate(user_id="u1", job_posting="Senior Python Engineer with AWS")

    # The orchestrator produced a scored draft and the workflow state is whole.
    assert state.job is not None
    assert state.candidate_profile is not None
    assert state.evidence is not None
    assert state.current_draft is not None
    assert state.current_trust is not None
    assert state.current_ats is not None
    # Scripted to pass on the first attempt.
    assert state.passed is True
    assert state.iteration == 0


def test_generate_reusesCachedCandidateProfileAcrossCalls() -> None:
    app = _app_with_scripted_calls(
        _FULL_GENERATION,  # generate() #1 — no cache yet, computes the profile
        _CACHED_PROFILE_GENERATION,  # generate() #2 — cache hit, no CP call
        _FULL_GENERATION,  # generate() #3 — re-ingest flagged it stale, recomputes
    )
    app.ensure_user("Ada", user_id="u1")
    app.add_document(user_id="u1", filename="resume.txt", text="Built Python services on AWS.")

    app.generate(user_id="u1", job_posting="Backend Engineer")
    first = app._candidate_profiles.get("u1")
    assert first is not None and first.stale is False

    # A second generation against a different job must not recompute the
    # candidate profile — it's job-independent and nothing was re-ingested.
    app.generate(user_id="u1", job_posting="Data Engineer")
    second = app._candidate_profiles.get("u1")
    assert second is not None
    assert second.updated_at == first.updated_at

    # Re-ingesting a document flags the cache stale; the next generation
    # recomputes and bumps updated_at.
    app.add_document(user_id="u1", filename="resume2.txt", text="Also led a team of five.")
    app.generate(user_id="u1", job_posting="Data Engineer")
    third = app._candidate_profiles.get("u1")
    assert third is not None
    assert third.updated_at != first.updated_at


def test_persist_noScoredDraft_isNoOp(app: TrustResumeApp) -> None:
    """``_persist`` is a no-op for an unscored ``WorkflowState`` (e.g. a run that
    errored before writing any draft) — nothing to write, and it must not raise.
    """
    from trustresume.models import WorkflowState

    app.ensure_user("Ada", user_id="u1")
    empty_state = WorkflowState(user_id="u1")  # no drafts/trust_reports/ats_reports

    app._persist(empty_state)  # must not raise

    row_count = app._resumes._conn.execute(
        "SELECT COUNT(*) FROM generated_resumes WHERE user_id = ?", ("u1",)
    ).fetchone()[0]
    assert row_count == 0


def test_generate_isolatesUsers(app: TrustResumeApp) -> None:
    # User u2 has no documents; generation still works but retrieves nothing
    # of u1's — verifying isolation carries through the full pipeline.
    app.ensure_user("Ada", user_id="u1")
    app.ensure_user("Bob", user_id="u2")
    app.add_document(user_id="u1", filename="r.txt", text="Secret Python work for u1")

    state = app.generate(user_id="u2", job_posting="Python role")
    assert state.evidence is not None
    assert state.evidence.chunks == []  # u2 sees none of u1's evidence


# --- job CRUD, job-scoped documents, generate_for_job, resume export -------


def test_createJob_extractsAndPersists() -> None:
    app = _app_with_scripted_calls([_JOB_DESCRIPTION_CALL])
    app.ensure_user("Ada", user_id="u1")

    row = app.create_job(user_id="u1", job_posting="Need a Senior Python Engineer")

    assert row["title"] == "Senior Python Engineer"
    assert row["summary"] == "Senior Python Engineer"  # title only, no company scripted


def test_getJob_listJobs_ownershipScoped() -> None:
    app = _app_with_scripted_calls([_JOB_DESCRIPTION_CALL])
    app.ensure_user("Ada", user_id="u1")
    app.ensure_user("Bob", user_id="u2")
    row = app.create_job(user_id="u1", job_posting="Need a Senior Python Engineer")
    job_id = row["id"]

    assert app.get_job(user_id="u1", job_id=job_id) is not None
    assert app.get_job(user_id="u2", job_id=job_id) is None
    assert len(app.list_jobs("u1")) == 1
    assert app.list_jobs("u2") == []


def test_updateJob_reExtractsAndReturnsUpdatedRow_unownedReturnsNoneWithoutExtraction() -> None:
    app = _app_with_scripted_calls([_JOB_DESCRIPTION_CALL, _JOB_DESCRIPTION_CALL])
    app.ensure_user("Ada", user_id="u1")
    app.ensure_user("Bob", user_id="u2")
    row = app.create_job(user_id="u1", job_posting="Need a Senior Python Engineer")
    job_id = row["id"]

    # Unowned update must be checked BEFORE extraction — consumes none of the
    # remaining scripted LLM calls if it short-circuits correctly.
    assert app.update_job(user_id="u2", job_id=job_id, job_posting="anything") is None

    updated = app.update_job(
        user_id="u1", job_id=job_id, job_posting="Need a Senior Python Engineer"
    )
    assert updated is not None
    assert updated["title"] == "Senior Python Engineer"


def test_deleteJob_removesJob_pastResumesKeptWithJobIdNulled() -> None:
    app = _app_with_scripted_calls(_FULL_GENERATION)
    app.ensure_user("Ada", user_id="u1")
    app.add_document(user_id="u1", filename="resume.txt", text="Built Python services on AWS.")
    job_row = app.create_job(user_id="u1", job_posting="Senior Python Engineer role")
    job_id = job_row["id"]

    state = app.generate_for_job(user_id="u1", job_id=job_id)
    assert state is not None
    resume_id = state.resume_id
    assert resume_id is not None

    assert app.delete_job(user_id="u1", job_id=job_id) is True
    assert app.delete_job(user_id="u1", job_id=job_id) is False  # already gone
    assert app.get_job(user_id="u1", job_id=job_id) is None

    resume_row = app.get_resume(user_id="u1", resume_id=resume_id)
    assert resume_row is not None
    assert resume_row["job_id"] is None
    assert resume_row["job_title"] == "Senior Python Engineer"


def test_linkDocumentToJob_requiresOwnershipOfBothSides() -> None:
    app = _app_with_scripted_calls([_JOB_DESCRIPTION_CALL])
    app.ensure_user("Ada", user_id="u1")
    app.ensure_user("Bob", user_id="u2")
    job_row = app.create_job(user_id="u1", job_posting="role")
    doc_id = app.add_document(user_id="u1", filename="r.txt", text="Some content.")

    assert app.link_document_to_job(user_id="u1", job_id=job_row["id"], document_id=doc_id) is True
    assert app.link_document_to_job(user_id="u2", job_id=job_row["id"], document_id=doc_id) is False
    assert app.link_document_to_job(user_id="u1", job_id="missing", document_id=doc_id) is False
    assert (
        app.link_document_to_job(user_id="u1", job_id=job_row["id"], document_id="missing") is False
    )


def test_uploadDocumentForJob_linksInOneStep_unownedJobReturnsNone() -> None:
    app = _app_with_scripted_calls([_JOB_DESCRIPTION_CALL])
    app.ensure_user("Ada", user_id="u1")
    job_row = app.create_job(user_id="u1", job_posting="role")

    doc_id = app.upload_document_for_job(
        user_id="u1",
        job_id=job_row["id"],
        filename="r.txt",
        data=b"Built Python services.",
        document_type=DocumentType.RESUME,
    )
    assert doc_id is not None

    assert (
        app.upload_document_for_job(
            user_id="u1", job_id="missing", filename="x.txt", data=b"content"
        )
        is None
    )


def test_listDocumentsForJob_genericPoolUnionedWithLinked_unownedJobReturnsNone() -> None:
    app = _app_with_scripted_calls([_JOB_DESCRIPTION_CALL])
    app.ensure_user("Ada", user_id="u1")
    job_row = app.create_job(user_id="u1", job_posting="role")
    generic_id = app.add_document(user_id="u1", filename="generic.txt", text="Generic content.")
    linked_id = app.upload_document_for_job(
        user_id="u1", job_id=job_row["id"], filename="linked.txt", data=b"Linked content."
    )

    docs = app.list_documents_for_job(user_id="u1", job_id=job_row["id"])
    assert docs is not None
    assert {d["id"] for d in docs} == {generic_id, linked_id}
    assert app.list_documents_for_job(user_id="u2", job_id=job_row["id"]) is None


def test_generateForJob_scopesRetrievalAndSkipsReExtraction_unownedReturnsNone() -> None:
    # Only CP/Draft/Trust scripted — no second JobDescription call, since
    # generate_for_job re-uses the job's already-extracted JobDescription.
    app = _app_with_scripted_calls(
        [_JOB_DESCRIPTION_CALL], [_CANDIDATE_PROFILE_CALL, _RESUME_DRAFT_CALL, _TRUST_CALL]
    )
    app.ensure_user("Ada", user_id="u1")
    job_row = app.create_job(user_id="u1", job_posting="Senior Python Engineer role")
    app.upload_document_for_job(
        user_id="u1",
        job_id=job_row["id"],
        filename="resume.txt",
        data=b"Built Python services on AWS.",
    )

    state = app.generate_for_job(user_id="u1", job_id=job_row["id"])

    assert state is not None
    assert state.job_id == job_row["id"]
    assert state.resume_id is not None
    assert app.generate_for_job(user_id="u2", job_id=job_row["id"]) is None


def test_getResume_listResumesForJob_ownershipScoped() -> None:
    app = _app_with_scripted_calls(_FULL_GENERATION)
    app.ensure_user("Ada", user_id="u1")
    app.add_document(user_id="u1", filename="resume.txt", text="Built Python services on AWS.")
    job_row = app.create_job(user_id="u1", job_posting="role")
    state = app.generate_for_job(user_id="u1", job_id=job_row["id"])
    assert state is not None

    assert app.get_resume(user_id="u1", resume_id=state.resume_id) is not None  # type: ignore[arg-type]
    assert app.get_resume(user_id="u2", resume_id=state.resume_id) is None  # type: ignore[arg-type]

    resumes = app.list_resumes_for_job(user_id="u1", job_id=job_row["id"])
    assert resumes is not None
    assert len(resumes) == 1
    assert app.list_resumes_for_job(user_id="u2", job_id=job_row["id"]) is None


def test_persist_rendersExportsUnconditionally_andRejectionDataOnlyWhenFailed() -> None:
    from trustresume.models import (
        ATSReport,
        QualityGate,
        ResumeDraft,
        ResumeSection,
        TrustReport,
        WorkflowState,
    )

    app = _app_with_scripted_calls([])
    app.ensure_user("Ada", user_id="u1")

    passing_state = WorkflowState(
        user_id="u1",
        gate=QualityGate(),
        drafts=[
            ResumeDraft(summary="s", sections=[ResumeSection(heading="Skills", bullets=["x"])])
        ],
        trust_reports=[TrustReport(claims=[], score=95.0)],
        ats_reports=[ATSReport(score=90.0)],
    )
    app._persist(passing_state)
    passing_row = app.get_resume(user_id="u1", resume_id=passing_state.resume_id)  # type: ignore[arg-type]
    assert passing_row is not None
    assert bytes(passing_row["pdf_bytes"])[:5] == b"%PDF-"
    assert passing_row["markdown_text"]
    assert passing_row["rejection_reason"] is None
    assert passing_row["improvement_suggestions"] is None

    failing_state = WorkflowState(
        user_id="u1",
        gate=QualityGate(),
        drafts=[ResumeDraft(summary="s", sections=[], iteration=3)],
        trust_reports=[TrustReport(claims=[], score=62.0, iteration=3)],
        ats_reports=[ATSReport(score=78.0, iteration=3)],
        iteration=3,
    )
    app._persist(failing_state)
    failing_row = app.get_resume(user_id="u1", resume_id=failing_state.resume_id)  # type: ignore[arg-type]
    assert failing_row is not None
    assert bytes(failing_row["pdf_bytes"])[:5] == b"%PDF-"
    assert failing_row["rejection_reason"] is not None
    assert failing_row["improvement_suggestions"] is not None
