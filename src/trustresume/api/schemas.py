"""HTTP request/response schemas for the API.

Distinct from the internal ``trustresume.models`` domain types: these shape
what crosses the wire to the React client. Keeping them separate means the
domain model can change without silently altering the public API contract.

Milestone M7 (React frontend + FastAPI backend).
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from trustresume.models import DocumentType, EvidenceSet, ResumeDraft, RunUsage, WorkflowState

# ``min_length=1`` alone accepts a whitespace-only string (e.g. " "), which
# would sail through validation and then silently produce a document with
# zero chunks, or a job/search query with nothing to extract or match.
# ``strip_whitespace=True`` runs before the length check, so a whitespace-only
# value has length 0 by the time it's checked — every user-supplied text
# field below uses this, not a bare ``Field(..., min_length=1)``.
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class AddDocumentRequest(BaseModel):
    """Body for ingesting a document as text (React sends file contents)."""

    filename: NonEmptyStr
    text: NonEmptyStr
    document_type: DocumentType = DocumentType.OTHER


class DocumentSummary(BaseModel):
    """One stored document, for listing in the UI."""

    id: str
    filename: str
    document_type: str


class GenerateRequest(BaseModel):
    """Body for a resume generation run."""

    job_posting: NonEmptyStr


class SearchRequest(BaseModel):
    """Body for an ad-hoc evidence search (outside a full generation run)."""

    query: NonEmptyStr
    limit: int = Field(default=5, ge=1, le=50)


class EvidenceChunkView(BaseModel):
    """One retrieved evidence chunk, for the search results view."""

    chunk_id: str
    document_type: str
    source_document: str | None
    text: str
    score: float | None


class SearchResponse(BaseModel):
    """Ranked evidence chunks for a search query."""

    query: str
    chunks: list[EvidenceChunkView]

    @classmethod
    def from_evidence(cls, evidence: EvidenceSet) -> SearchResponse:
        return cls(
            query=evidence.query,
            chunks=[
                EvidenceChunkView(
                    chunk_id=c.chunk_id,
                    document_type=c.document_type.value,
                    source_document=c.source_document,
                    text=c.text,
                    score=c.score,
                )
                for c in evidence.chunks
            ],
        )


class ClaimView(BaseModel):
    """A flagged unsupported claim, for the trust panel."""

    text: str
    category: str


class UsageView(BaseModel):
    """What a run consumed, flattened for the wire.

    Flat scalars rather than the nested :class:`~trustresume.models.RunUsage`
    (which carries per-model and per-node breakdowns): a client showing "this
    took 12s and cost $0.04" shouldn't have to sum arrays, and the detailed
    breakdown stays available server-side in logs and the eval harness.
    """

    llm_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    duration_ms: float
    cost_usd: float | None = Field(
        None, description="None when any model used has no configured price."
    )

    @classmethod
    def from_usage(cls, usage: RunUsage) -> UsageView:
        """Project a :class:`RunUsage` onto the wire shape."""
        return cls(
            llm_calls=usage.llm_calls,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            duration_ms=round(usage.total_duration_ms, 1),
            cost_usd=usage.cost_usd,
        )


class GenerateResponse(BaseModel):
    """The result of a generation run, shaped for the React client.

    Surfaces the *real* scores and the pass/fail even when the shipped draft
    didn't pass — the UI decides how to warn. ``exhausted`` is currently
    always ``True`` for any real run: the quality loop no longer exits early
    on a pass, so every completed run has reached the iteration cap by the
    time this is built. Kept rather than removed since it's still a
    meaningful concept on hand-built ``WorkflowState``s (e.g. in tests).
    """

    draft: ResumeDraft
    trust_score: float
    ats_score: float
    passed: bool
    exhausted: bool
    iterations: int
    hallucinations: list[ClaimView]
    missing_keywords: list[str]
    resume_id: str | None = None
    usage: UsageView | None = None

    @classmethod
    def from_state(cls, state: WorkflowState) -> GenerateResponse:
        """Project a completed :class:`WorkflowState` into the API response."""
        draft = state.final_draft
        trust = state.final_trust
        ats = state.final_ats
        if draft is None or trust is None or ats is None:
            raise ValueError("workflow produced no scored draft")
        return cls(
            draft=draft,
            trust_score=trust.score,
            ats_score=ats.score,
            passed=state.final_passed,
            exhausted=state.is_exhausted,
            iterations=state.iteration,
            hallucinations=[
                ClaimView(text=c.text, category=c.category.value) for c in trust.hallucinations
            ],
            missing_keywords=ats.missing_keywords,
            resume_id=state.resume_id,
            usage=UsageView.from_usage(state.usage) if state.usage else None,
        )


class CreateJobRequest(BaseModel):
    """Body for creating (or replacing, via PUT) a job."""

    job_posting: NonEmptyStr


class JobSummary(BaseModel):
    """One stored job, for listing."""

    id: str
    title: str | None
    company: str | None
    summary: str | None
    created_at: str


class JobDetail(JobSummary):
    """A single job's full extracted detail."""

    raw_posting: str
    seniority: str
    required_skills: list[str]
    preferred_skills: list[str]
    responsibilities: list[str]
    keywords: list[str]


class ResumeSummary(BaseModel):
    """One stored resume, for listing (no draft/export payload)."""

    id: str
    job_id: str | None
    job_title: str | None
    iteration: int
    trust_score: float
    ats_score: float
    passed: bool
    created_at: str


class ResumeDetail(ResumeSummary):
    """A single resume's full stored detail, including the draft itself."""

    draft: ResumeDraft
    rejection_reason: str | None
    improvement_suggestions: str | None
    usage: UsageView | None = Field(
        None, description="What this resume cost to generate; None for rows written unmeasured."
    )
