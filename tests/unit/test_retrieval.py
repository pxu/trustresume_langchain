"""Unit tests for the Chroma vector store and embeddings.

Runs entirely in-process: ``chromadb.EphemeralClient()`` plus the
deterministic ``fake_embedder`` fixture. The key thing under test is the
ADR-0001 isolation guarantee — a search filtered by ``user_id`` never returns
another user's data.
"""

from __future__ import annotations

import uuid

import chromadb
import pytest
from langchain_core.embeddings import Embeddings

from tests.fakes import FakeEmbeddings
from trustresume.models import DocumentType, EvidenceChunk
from trustresume.retrieval import ChromaVectorStore
from trustresume.retrieval.embedder import FastEmbedEmbeddings


@pytest.fixture
def store(fake_embedder: FakeEmbeddings) -> ChromaVectorStore:
    # chromadb.EphemeralClient() caches its underlying storage by settings
    # hash, so distinct client instances with the same collection name share
    # state within a process. A unique collection name per test keeps tests
    # isolated; production code always uses one long-lived collection name,
    # so this only affects the test fixture.
    return ChromaVectorStore(
        chromadb.EphemeralClient(), fake_embedder, collection_name=f"test-{uuid.uuid4().hex}"
    )


def _chunk(chunk_id: str, user_id: str, text: str) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        user_id=user_id,
        document_id="d1",
        document_type=DocumentType.RESUME,
        source_document="resume.pdf",
        text=text,
    )


def test_fakeEmbeddings_isEmbeddings(fake_embedder: FakeEmbeddings) -> None:
    assert isinstance(fake_embedder, Embeddings)
    assert isinstance(FastEmbedEmbeddings(), Embeddings)


def test_fakeEmbeddings_deterministicAndSized(fake_embedder: FakeEmbeddings) -> None:
    a = fake_embedder.embed_documents(["hello"])
    b = fake_embedder.embed_documents(["hello"])
    assert a == b
    assert len(a[0]) == fake_embedder.dimension


def test_search_returnsOwnChunks(store: ChromaVectorStore) -> None:
    store.upsert_chunks(
        [
            _chunk("c1", "u1", "Python and AWS backend experience"),
            _chunk("c2", "u1", "Led a team of five engineers"),
        ]
    )

    result = store.search(user_id="u1", query="Python and AWS backend experience", limit=2)
    assert result.user_id == "u1"
    assert {c.chunk_id for c in result.chunks} == {"c1", "c2"}
    # Payload survived the round trip.
    top = result.chunks[0]
    assert top.document_type is DocumentType.RESUME
    assert top.source_document == "resume.pdf"
    assert top.score is not None


def test_search_isolatesUsers(store: ChromaVectorStore) -> None:
    store.upsert_chunks([_chunk("c1", "u1", "secret evidence for user one")])
    store.upsert_chunks([_chunk("c2", "u2", "secret evidence for user two")])

    result = store.search(user_id="u2", query="secret evidence", limit=10)
    assert [c.chunk_id for c in result.chunks] == ["c2"]


def test_search_emptyCollection_returnsEmptySet(store: ChromaVectorStore) -> None:
    result = store.search(user_id="u1", query="anything")
    assert result.chunks == []


def test_upsert_emptyList_isNoOp(store: ChromaVectorStore) -> None:
    store.upsert_chunks([])  # must not raise
    assert store.search(user_id="u1", query="x").chunks == []


def test_deleteChunks_removesPoints(store: ChromaVectorStore) -> None:
    store.upsert_chunks([_chunk("c1", "u1", "removable"), _chunk("c2", "u1", "keeper")])
    store.delete_chunks(["c1"])

    remaining = {c.chunk_id for c in store.search(user_id="u1", query="anything", limit=10).chunks}
    assert remaining == {"c2"}
