"""Unit tests for CandidateProfileService — the cache-check wrapper around
CandidateProfileAgent.

Uses a fake agent (spy) rather than a scripted chat model: these tests are
about the caching control flow (call it once, reuse the cache, recompute when
stale), not LLM extraction correctness — that's covered by
``test_candidateProfileAgent_structuresCandidateText`` in ``test_agents.py``.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Awaitable
from typing import TypeVar

import chromadb

from tests.fakes import FakeEmbeddings
from trustresume.ingestion import IngestionService
from trustresume.models import CandidateProfile
from trustresume.orchestration import CandidateProfileService
from trustresume.retrieval import ChromaVectorStore
from trustresume.storage import (
    CandidateProfileRepository,
    ChunkRepository,
    DocumentRepository,
    UserRepository,
)

_T = TypeVar("_T")


def run(awaitable: Awaitable[_T]) -> _T:
    return asyncio.run(awaitable)  # type: ignore[arg-type]


class FakeCandidateProfileAgent:
    """Records how many times it was called; returns a scripted profile."""

    def __init__(self, profile: CandidateProfile) -> None:
        self.calls = 0
        self.profile = profile

    async def run(self, candidate_text: str) -> CandidateProfile:
        self.calls += 1
        return self.profile


def _make(
    db: sqlite3.Connection,
) -> tuple[CandidateProfileService, FakeCandidateProfileAgent, ChunkRepository]:
    UserRepository(db).create("A", user_id="u1")
    chunks = ChunkRepository(db)
    agent = FakeCandidateProfileAgent(CandidateProfile(name="Jordan Rivera", skills=["python"]))
    service = CandidateProfileService(
        agent=agent,  # type: ignore[arg-type]
        chunks=chunks,
        profiles=CandidateProfileRepository(db),
    )
    return service, agent, chunks


def test_getOrRefresh_noCacheYet_computesAndPersists(db: sqlite3.Connection) -> None:
    service, agent, _chunks = _make(db)

    profile = run(service.get_or_refresh("u1"))

    assert profile.name == "Jordan Rivera"
    assert agent.calls == 1


def test_getOrRefresh_documentsUnchanged_reusesCacheWithoutRecomputing(
    db: sqlite3.Connection,
) -> None:
    service, agent, _chunks = _make(db)
    run(service.get_or_refresh("u1"))
    assert agent.calls == 1

    profile = run(service.get_or_refresh("u1"))

    assert profile.name == "Jordan Rivera"
    assert agent.calls == 1  # cache hit — no second LLM call


def test_getOrRefresh_documentsChanged_recomputesAndClearsStaleFlag(
    db: sqlite3.Connection,
) -> None:
    service, agent, _chunks = _make(db)
    run(service.get_or_refresh("u1"))
    assert agent.calls == 1

    # Simulate what IngestionService does after a document mutation.
    service._profiles.mark_stale("u1")
    agent.profile = CandidateProfile(name="Jordan Rivera", skills=["python", "aws"])

    profile = run(service.get_or_refresh("u1"))

    assert profile.skills == ["python", "aws"]
    assert agent.calls == 2
    # Recompute clears the flag — a third call is a cache hit again.
    run(service.get_or_refresh("u1"))
    assert agent.calls == 2


def test_ingestionMutation_invalidatesCache_endToEnd(
    db: sqlite3.Connection, fake_embedder: FakeEmbeddings
) -> None:
    """The real IngestionService -> stale flag -> service-refresh chain."""
    service, agent, chunks = _make(db)
    ingestion = IngestionService(
        documents=DocumentRepository(db),
        chunks=chunks,
        vector_store=ChromaVectorStore(
            chromadb.EphemeralClient(), fake_embedder, collection_name="test-cps"
        ),
        candidate_profiles=service._profiles,
    )
    ingestion.ingest_text(user_id="u1", filename="r.txt", text="Python and AWS work.")

    run(service.get_or_refresh("u1"))
    assert agent.calls == 1
    run(service.get_or_refresh("u1"))
    assert agent.calls == 1  # still cached — ingestion hasn't mutated since

    ingestion.ingest_text(user_id="u1", filename="r2.txt", text="More experience.")
    run(service.get_or_refresh("u1"))

    assert agent.calls == 2  # new document flagged the cache stale
