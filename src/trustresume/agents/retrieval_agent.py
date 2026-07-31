"""Evidence Retrieval agent — job description -> retrieved ``EvidenceSet``.

Second step in the pipeline (ADR-0002). Unlike the others this is deliberately
*not* an LLM agent: retrieval is deterministic search over the user's own
documents, and dressing it up as an LLM call would only add nondeterminism
and cost. It builds a query from the structured job fields (via
``retrieval.query.build_query``) and delegates to a user-scoped retriever
(vector-only ``ChromaVectorStore``, or hybrid vector+keyword
``HybridRetriever``), so isolation (ADR-0001) is enforced downstream.

Keeping it behind the same ``run(...)`` shape as the LLM agents lets the
orchestrator treat all agents uniformly.

Milestone M4 (agents); hybrid retrieval added post-M4.
"""

from __future__ import annotations

from typing import Protocol

from trustresume.models import EvidenceSet, JobDescription
from trustresume.retrieval.query import build_query


class _Retriever(Protocol):
    """What ``EvidenceRetrievalAgent`` needs — ``ChromaVectorStore`` and
    ``HybridRetriever`` both satisfy this without either depending on the other.
    """

    def search(
        self, *, user_id: str, query: str, limit: int, document_ids: list[str] | None = None
    ) -> EvidenceSet: ...


class EvidenceRetrievalAgent:
    """Retrieves evidence relevant to a job description for one user."""

    def __init__(self, retriever: _Retriever, *, top_k: int = 8) -> None:
        self._retriever = retriever
        self._top_k = top_k

    async def run(
        self,
        *,
        user_id: str,
        job: JobDescription,
        document_ids: list[str] | None = None,
    ) -> EvidenceSet:
        """Retrieve the top-k evidence chunks for this user and job.

        ``document_ids``, when given, scopes retrieval to that set (job-scoped
        retrieval) — resolved by the caller (e.g. the orchestrator, via
        ``DocumentRepository.list_eligible_document_ids``), not by this agent:
        keeps the agent's own dependency surface retrieval-only, with no new
        SQLite/DocumentRepository dependency of its own.
        """
        query = build_query(job)
        return self._retriever.search(
            user_id=user_id, query=query, limit=self._top_k, document_ids=document_ids
        )
