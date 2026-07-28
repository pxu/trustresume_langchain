"""CandidateProfileService — cached, job-independent candidate profile resolution.

Wraps :class:`~trustresume.agents.CandidateProfileAgent` (a pure LLM
extraction step) with the deterministic cache-check and document-assembly
work around it. A candidate's profile doesn't change per job, so recomputing
it with an LLM call on every ``generate()`` would be wasted cost for no new
information — it's recomputed only when the cache is missing or has been
flagged stale by ``IngestionService`` after a document upload/delete.

This is deterministic, not an LLM reasoning step itself (the LLM call it
sometimes makes lives entirely inside the wrapped agent), so like
``IngestionService`` it's named and treated as a **service**, not an agent —
see "Ingestion is a separate write path" in
``architecture/high-level-design.md`` for the Service-vs-Agent distinction
this follows.

Milestone M5 (orchestration).
"""

from __future__ import annotations

import hashlib

from trustresume.agents import CandidateProfileAgent
from trustresume.models import CandidateProfile
from trustresume.storage import CandidateProfileRepository, ChunkRepository


class CandidateProfileService:
    """Resolves a user's :class:`CandidateProfile`, recomputing only when stale."""

    def __init__(
        self,
        *,
        agent: CandidateProfileAgent,
        chunks: ChunkRepository,
        profiles: CandidateProfileRepository,
    ) -> None:
        self._agent = agent
        self._chunks = chunks
        self._profiles = profiles

    async def get_or_refresh(self, user_id: str) -> CandidateProfile:
        """Return the cached profile if fresh; otherwise recompute and cache it."""
        cached = self._profiles.get(user_id)
        if cached is not None and not cached.stale:
            return cached.profile

        text = self._assemble_candidate_text(user_id)
        profile = await self._agent.run(text)
        self._profiles.upsert(user_id=user_id, profile=profile, doc_hash=self._hash(text))
        return profile

    def _assemble_candidate_text(self, user_id: str) -> str:
        """Concatenate all of a user's already-ingested chunks into one blob.

        Reuses ``ChunkRepository`` rather than re-reading raw uploads — by the
        time a generation runs, ingestion has already parsed/cleaned/chunked
        everything (see ``architecture/high-level-design.md``), so this is the
        only text source that's actually available and guaranteed current.
        Adjacent chunks may share a little overlap text (``chunker.py``'s
        overlap window); harmless for an LLM extraction, not worth
        deduplicating here.
        """
        rows = self._chunks.list_for_user(user_id)
        return "\n\n".join(row["text"] for row in rows)

    @staticmethod
    def _hash(text: str) -> str:
        """A content hash recorded for diagnostics, not for the cache decision.

        The cache gate is purely the ``stale`` flag (set by
        ``IngestionService``); this hash just lets a debugger later confirm
        whether the underlying text actually changed between two profiles.
        """
        return hashlib.sha256(text.encode()).hexdigest()
