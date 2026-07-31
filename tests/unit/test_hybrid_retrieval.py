"""Unit tests for hybrid (vector + keyword) retrieval (``retrieval.hybrid``).

``_reciprocal_rank_fusion`` is tested directly as a pure function (exact rank
math); ``HybridRetriever.search`` is tested against real collaborators (an
in-memory SQLite connection + ``chromadb.EphemeralClient()``) so the fusion
is exercised through the same two data stores production code uses, not a
mock standing in for either.
"""

from __future__ import annotations

import sqlite3
import uuid

import chromadb
import pytest

from tests.fakes import FakeEmbeddings
from trustresume.models import DocumentType, EvidenceChunk
from trustresume.retrieval import ChromaVectorStore, HybridRetriever
from trustresume.retrieval.hybrid import _reciprocal_rank_fusion
from trustresume.storage import ChunkRepository, DocumentRepository, UserRepository


def _chunk(chunk_id: str, *, score: float | None = None) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id, user_id="u1", document_id="d1", text=f"text {chunk_id}", score=score
    )


# --- _reciprocal_rank_fusion: pure function -------------------------------


def test_rrf_chunkInBothLists_outranksChunkInOnlyOne() -> None:
    # "b" is rank 2 in both lists; "a" is rank 1 in only the first.
    list_a = [_chunk("a"), _chunk("b")]
    list_b = [_chunk("c"), _chunk("b")]

    fused = _reciprocal_rank_fusion([list_a, list_b], k=60)

    assert [c.chunk_id for c in fused][0] == "b"


def test_rrf_singleList_preservesOrder() -> None:
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    fused = _reciprocal_rank_fusion([chunks], k=60)
    assert [c.chunk_id for c in fused] == ["a", "b", "c"]


def test_rrf_emptyLists_returnsEmpty() -> None:
    assert _reciprocal_rank_fusion([[], []], k=60) == []


def test_rrf_firstSeenChunkObjectWins() -> None:
    # "a" appears in both lists with different EvidenceChunk instances (e.g.
    # the Chroma version carries a similarity `score`, the FTS version
    # doesn't) — the richer, first-seen one should be kept.
    rich = _chunk("a", score=0.9)
    plain = _chunk("a", score=None)

    fused = _reciprocal_rank_fusion([[rich], [plain]], k=60)

    assert fused[0].score == 0.9


def test_rrf_scoreExactly_matchesFormula() -> None:
    # k=60, "a" at rank 1 in one list only: score = 1/(60+1).
    fused = _reciprocal_rank_fusion([[_chunk("a")]], k=60)
    assert fused[0].chunk_id == "a"
    # Verify via the internal score directly isn't exposed, so check ordering
    # against a second chunk whose combined score should be lower.
    list_a = [_chunk("a")]  # rank 1: 1/61
    list_b = [_chunk("x"), _chunk("b")]  # "b" rank 2: 1/62
    fused2 = _reciprocal_rank_fusion([list_a, list_b], k=60)
    assert fused2[0].chunk_id == "a"  # 1/61 > 1/62


# --- HybridRetriever: real Chroma + real SQLite ---------------------------


@pytest.fixture
def db() -> sqlite3.Connection:
    from trustresume.storage import connect, init_db

    conn = connect(":memory:")
    init_db(conn)
    return conn


@pytest.fixture
def retriever(db: sqlite3.Connection, fake_embedder: FakeEmbeddings) -> HybridRetriever:
    vectors = ChromaVectorStore(
        chromadb.EphemeralClient(), fake_embedder, collection_name=f"test-{uuid.uuid4().hex}"
    )
    return HybridRetriever(vectors, ChunkRepository(db))


def _ingest(db: sqlite3.Connection, retriever: HybridRetriever, chunk: EvidenceChunk) -> None:
    """Write a chunk to both stores, exactly as IngestionService does."""
    ChunkRepository(db).add(chunk, chunk_index=0)
    retriever._vectors.upsert_chunks([chunk])  # noqa: SLF001 — test-only direct write


def test_hybridRetriever_keywordExactMatch_surfacesEvenIfVectorMisses(
    db: sqlite3.Connection, retriever: HybridRetriever
) -> None:
    """The core case hybrid retrieval exists for: a chunk containing the
    exact query term should surface even when the hash-based fake embedder
    (which carries no real semantic signal) ranks it poorly for vector search.
    """
    UserRepository(db).create("A", user_id="u1")
    doc_id = DocumentRepository(db).create(
        user_id="u1", filename="r.txt", document_type=DocumentType.RESUME, content_hash="h1"
    )
    target = EvidenceChunk(
        chunk_id="c-lambda",
        user_id="u1",
        document_id=doc_id,
        text="Built serverless functions with AWS Lambda and API Gateway",
    )
    _ingest(db, retriever, target)
    for i in range(5):
        _ingest(
            db,
            retriever,
            EvidenceChunk(
                chunk_id=f"c-filler-{i}",
                user_id="u1",
                document_id=doc_id,
                text=f"Unrelated filler content number {i}",
            ),
        )

    result = retriever.search(user_id="u1", query="Lambda", limit=5)

    assert "c-lambda" in {c.chunk_id for c in result.chunks}


def test_hybridRetriever_isolatesUsers(db: sqlite3.Connection, retriever: HybridRetriever) -> None:
    UserRepository(db).create("A", user_id="u1")
    UserRepository(db).create("B", user_id="u2")
    doc_id = DocumentRepository(db).create(
        user_id="u1", filename="r.txt", document_type=DocumentType.RESUME, content_hash="h1"
    )
    _ingest(
        db,
        retriever,
        EvidenceChunk(chunk_id="c1", user_id="u1", document_id=doc_id, text="secret Kubernetes"),
    )

    result = retriever.search(user_id="u2", query="Kubernetes", limit=5)

    assert result.chunks == []


def test_hybridRetriever_documentIdsFilter_threadsToBothVectorAndKeywordSearch(
    db: sqlite3.Connection, retriever: HybridRetriever
) -> None:
    """document_ids must scope both halves of the fusion — a chunk from an
    ineligible document must not surface via the keyword side even if it
    would win on vector similarity (or vice versa).
    """
    UserRepository(db).create("A", user_id="u1")
    docs = DocumentRepository(db)
    doc1 = docs.create(
        user_id="u1", filename="a.txt", document_type=DocumentType.RESUME, content_hash="h1"
    )
    doc2 = docs.create(
        user_id="u1", filename="b.txt", document_type=DocumentType.RESUME, content_hash="h2"
    )
    _ingest(
        db,
        retriever,
        EvidenceChunk(chunk_id="c1", user_id="u1", document_id=doc1, text="Kubernetes expert"),
    )
    _ingest(
        db,
        retriever,
        EvidenceChunk(chunk_id="c2", user_id="u1", document_id=doc2, text="Kubernetes expert too"),
    )

    scoped = retriever.search(user_id="u1", query="Kubernetes", limit=10, document_ids=[doc1])
    empty = retriever.search(user_id="u1", query="Kubernetes", limit=10, document_ids=[])

    assert [c.chunk_id for c in scoped.chunks] == ["c1"]
    assert empty.chunks == []


def test_hybridRetriever_respectsLimit(db: sqlite3.Connection, retriever: HybridRetriever) -> None:
    UserRepository(db).create("A", user_id="u1")
    doc_id = DocumentRepository(db).create(
        user_id="u1", filename="r.txt", document_type=DocumentType.RESUME, content_hash="h1"
    )
    for i in range(10):
        _ingest(
            db,
            retriever,
            EvidenceChunk(
                chunk_id=f"c{i}", user_id="u1", document_id=doc_id, text=f"Python engineer {i}"
            ),
        )

    result = retriever.search(user_id="u1", query="Python engineer", limit=3)

    assert len(result.chunks) == 3


def test_hybridRetriever_noMatches_returnsEmptySet(
    db: sqlite3.Connection, retriever: HybridRetriever
) -> None:
    UserRepository(db).create("A", user_id="u1")
    result = retriever.search(user_id="u1", query="anything", limit=5)
    assert result.user_id == "u1"
    assert result.chunks == []
