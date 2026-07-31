"""Unit tests for the Chroma vector store and embeddings.

Runs entirely in-process: ``chromadb.EphemeralClient()`` plus the
deterministic ``fake_embedder`` fixture. The key thing under test is the
ADR-0001 isolation guarantee — a search filtered by ``user_id`` never returns
another user's data.
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

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


def test_fastEmbedEmbeddings_lazyLoadsModelOnFirstUse() -> None:
    """The underlying ``fastembed`` model is mocked out here (a real load can
    trigger a model download — this file stays offline per NFR-5); the real
    model is exercised by ``test_fastEmbedEmbeddings_realModel_embedsText``
    under the ``live`` marker instead. This test is about the *lazy-load*
    contract: no model construction until the first ``embed_*`` call.
    """
    with patch("fastembed.TextEmbedding") as mock_cls:
        mock_cls.return_value.embed.return_value = [[0.1, 0.2, 0.3]]

        embedder = FastEmbedEmbeddings(model_name="some/model")
        mock_cls.assert_not_called()  # constructing the wrapper loads nothing

        vectors = embedder.embed_documents(["hello"])

        mock_cls.assert_called_once_with(model_name="some/model")
        assert vectors == [[0.1, 0.2, 0.3]]

        embedder.embed_query("hello again")
        mock_cls.assert_called_once()  # second call reuses the cached model


def test_fastEmbedEmbeddings_embedQuery_delegatesToEmbedDocuments() -> None:
    with patch("fastembed.TextEmbedding") as mock_cls:
        mock_cls.return_value.embed.return_value = [[1.0, 2.0]]
        embedder = FastEmbedEmbeddings()

        assert embedder.embed_query("hello") == [1.0, 2.0]


@pytest.mark.live
def test_fastEmbedEmbeddings_realModel_embedsText() -> None:
    """Loads the real fastembed model — may download it on first run; that's
    exactly why this is ``live``-marked (deselected by default, per NFR-5)."""
    embedder = FastEmbedEmbeddings()
    vectors = embedder.embed_documents(["Built Python services on AWS."])
    assert len(vectors) == 1
    assert len(vectors[0]) == 384  # BAAI/bge-small-en-v1.5's dimension
    assert all(isinstance(v, float) for v in vectors[0])


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


def test_search_documentIdsFilter_restrictsToThoseDocuments(store: ChromaVectorStore) -> None:
    c1 = EvidenceChunk(
        chunk_id="c1",
        user_id="u1",
        document_id="d1",
        document_type=DocumentType.RESUME,
        text="Kubernetes expert here",
    )
    c2 = EvidenceChunk(
        chunk_id="c2",
        user_id="u1",
        document_id="d2",
        document_type=DocumentType.RESUME,
        text="Kubernetes expert here too",
    )
    store.upsert_chunks([c1, c2])

    scoped = store.search(user_id="u1", query="Kubernetes", limit=10, document_ids=["d1"])

    assert [c.chunk_id for c in scoped.chunks] == ["c1"]


def test_search_documentIdsFilter_stillScopedByUser(store: ChromaVectorStore) -> None:
    """document_ids alone must not bypass user isolation — the $and filter
    combines both conditions, not either.
    """
    store.upsert_chunks([_chunk("c1", "u1", "Kubernetes expert")])

    result = store.search(user_id="u2", query="Kubernetes", limit=10, document_ids=["d1"])

    assert result.chunks == []


def test_search_emptyDocumentIdsList_shortCircuitsWithoutCallingChroma(
    store: ChromaVectorStore,
) -> None:
    """An empty document_ids list must return empty without even querying
    Chroma — an empty $in list is itself invalid there (verified separately),
    not merely "matches nothing."
    """
    store.upsert_chunks([_chunk("c1", "u1", "Kubernetes expert")])

    result = store.search(user_id="u1", query="Kubernetes", limit=10, document_ids=[])

    assert result.chunks == []


def test_upsert_emptyList_isNoOp(store: ChromaVectorStore) -> None:
    store.upsert_chunks([])  # must not raise
    assert store.search(user_id="u1", query="x").chunks == []


def test_deleteChunks_removesPoints(store: ChromaVectorStore) -> None:
    store.upsert_chunks([_chunk("c1", "u1", "removable"), _chunk("c2", "u1", "keeper")])
    store.delete_chunks(["c1"])

    remaining = {c.chunk_id for c in store.search(user_id="u1", query="anything", limit=10).chunks}
    assert remaining == {"c2"}


def test_deleteChunks_emptyList_isNoOp(store: ChromaVectorStore) -> None:
    store.upsert_chunks([_chunk("c1", "u1", "keeper")])
    store.delete_chunks([])  # must not raise

    remaining = {c.chunk_id for c in store.search(user_id="u1", query="anything", limit=10).chunks}
    assert remaining == {"c1"}
