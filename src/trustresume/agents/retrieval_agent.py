"""Evidence Retrieval agent — job description -> retrieved ``EvidenceSet``.

Second step in the pipeline (ADR-0002). Unlike the others this is deliberately
*not* an LLM agent: retrieval is deterministic semantic search over the user's
own documents, and dressing it up as an LLM call would only add nondeterminism
and cost. It builds a query from the structured job fields (via
``retrieval.query.build_query``) and delegates to the user-scoped vector
store, so isolation (ADR-0001) is enforced downstream.

Keeping it behind the same ``run(...)`` shape as the LLM agents lets the
orchestrator treat all agents uniformly.

Milestone M4 (agents).
"""

from __future__ import annotations

from trustresume.models import EvidenceSet, JobDescription
from trustresume.retrieval import ChromaVectorStore
from trustresume.retrieval.query import build_query


class EvidenceRetrievalAgent:
    """Retrieves evidence relevant to a job description for one user."""

    def __init__(self, vector_store: ChromaVectorStore, *, top_k: int = 8) -> None:
        self._vectors = vector_store
        self._top_k = top_k

    async def run(self, *, user_id: str, job: JobDescription) -> EvidenceSet:
        """Retrieve the top-k evidence chunks for this user and job."""
        query = build_query(job)
        return self._vectors.search(user_id=user_id, query=query, limit=self._top_k)
