"""HTTP request/response schemas for the API.

Distinct from the internal ``trustresume.models`` domain types: these shape
what crosses the wire to the React client. Keeping them separate means the
domain model can change without silently altering the public API contract.

Milestone M7 (React frontend + FastAPI backend).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from trustresume.models import DocumentType, ResumeDraft, WorkflowState


class AddDocumentRequest(BaseModel):
    """Body for ingesting a document as text (React sends file contents)."""

    filename: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    document_type: DocumentType = DocumentType.OTHER


class DocumentSummary(BaseModel):
    """One stored document, for listing in the UI."""

    id: str
    filename: str
    document_type: str


class GenerateRequest(BaseModel):
    """Body for a resume generation run."""

    job_posting: str = Field(..., min_length=1)


class ClaimView(BaseModel):
    """A flagged unsupported claim, for the trust panel."""

    text: str
    category: str


class GenerateResponse(BaseModel):
    """The result of a generation run, shaped for the React client.

    Surfaces the *real* scores and the pass/fail even when the draft hit the
    iteration cap without passing (ADR-0005) — the UI decides how to warn.
    """

    draft: ResumeDraft
    trust_score: float
    ats_score: float
    passed: bool
    exhausted: bool
    iterations: int
    hallucinations: list[ClaimView]
    missing_keywords: list[str]

    @classmethod
    def from_state(cls, state: WorkflowState) -> GenerateResponse:
        """Project a completed :class:`WorkflowState` into the API response."""
        draft = state.current_draft
        trust = state.current_trust
        ats = state.current_ats
        if draft is None or trust is None or ats is None:
            raise ValueError("workflow produced no scored draft")
        return cls(
            draft=draft,
            trust_score=trust.score,
            ats_score=ats.score,
            passed=state.passed,
            exhausted=state.is_exhausted,
            iterations=state.iteration,
            hallucinations=[
                ClaimView(text=c.text, category=c.category.value) for c in trust.hallucinations
            ],
            missing_keywords=ats.missing_keywords,
        )
