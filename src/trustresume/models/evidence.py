"""Evidence retrieved from a candidate's own documents.

An ``EvidenceChunk`` is one embedded, retrievable slice of a source document.
It is the join point between the two stores (ADR-0001): the same chunk exists
as a Chroma vector (for semantic search) and as a SQLite metadata row, kept in
sync and both carrying ``user_id``. ``EvidenceSet`` is the Evidence Retrieval
agent's output for a single generation.

Milestone M1 (shared models).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .enums import DocumentType


class EvidenceChunk(BaseModel):
    """A single chunk of candidate evidence and its provenance.

    ``chunk_id`` is the shared identity across Chroma and SQLite — it doubles
    as the Chroma document id directly. ``score`` is the retrieval similarity
    for the query that surfaced it (``None`` when the chunk is being stored
    rather than returned from a search).
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(
        ..., min_length=1, description="Stable id shared by the Chroma document + SQLite row."
    )
    user_id: str = Field(
        ..., min_length=1, description="Owning user; every chunk is user-scoped (ADR-0001)."
    )
    document_id: str = Field(
        ..., min_length=1, description="Id of the source document this chunk came from."
    )
    document_type: DocumentType = Field(
        default=DocumentType.OTHER,
        description="Type of the source document.",
    )
    source_document: str | None = Field(
        None,
        description="Human-readable source filename/title, for citing evidence in reports.",
    )
    text: str = Field(..., min_length=1, description="The chunk's text content.")
    score: float | None = Field(
        None,
        description="Retrieval similarity score for the query that surfaced this chunk.",
    )


class EvidenceSet(BaseModel):
    """The evidence retrieved for one generation, scoped to a single user."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(..., min_length=1, description="User the evidence was retrieved for.")
    query: str = Field(
        ..., description="Query used to retrieve the evidence (typically job keywords)."
    )
    chunks: list[EvidenceChunk] = Field(
        default_factory=list,
        description="Retrieved chunks, conventionally ordered by descending relevance.",
    )
