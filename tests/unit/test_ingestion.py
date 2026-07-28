"""Unit tests for the ingestion pipeline.

Covers the pure steps (clean/chunk/parse) and the IngestionService's core
contract: chunks land in *both* stores, the stores stay in sync when embedding
fails, and delete removes from both.
"""

from __future__ import annotations

import sqlite3
import uuid

import chromadb
import pytest

from tests.fakes import FakeEmbeddings
from trustresume.ingestion import (
    IngestionService,
    UnsupportedDocumentError,
    chunk_text,
    clean_text,
    parse_document,
)
from trustresume.models import DocumentType
from trustresume.retrieval import ChromaVectorStore
from trustresume.storage import (
    CandidateProfileRepository,
    ChunkRepository,
    DocumentRepository,
    UserRepository,
)

# --- pure steps ------------------------------------------------------------


def test_cleanText_collapsesWhitespaceKeepsParagraphs() -> None:
    raw = "Hello   world\r\n\r\n\r\n  Second   line  \n\n"
    assert clean_text(raw) == "Hello world\nSecond line"


def test_chunkText_empty_returnsEmpty() -> None:
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunkText_groupsParagraphsUnderLimit() -> None:
    text = "para one\npara two\npara three"
    chunks = chunk_text(text, max_chars=100, overlap=0)
    # All fit comfortably, so they coalesce into a single chunk.
    assert chunks == ["para one\npara two\npara three"]


def test_chunkText_splitsWhenExceedingLimit() -> None:
    text = "aaaa\nbbbb\ncccc"
    chunks = chunk_text(text, max_chars=9, overlap=0)
    # "aaaa\nbbbb" = 9 chars fits; "cccc" starts a new chunk.
    assert chunks == ["aaaa\nbbbb", "cccc"]


def test_chunkText_hardSplitsOversizedParagraph() -> None:
    chunks = chunk_text("x" * 25, max_chars=10, overlap=2)
    assert all(len(c) <= 10 for c in chunks)
    assert len(chunks[0]) == 10  # first window is a full max_chars slice
    assert len(chunks) >= 3


@pytest.mark.parametrize("bad", [{"max_chars": 0}, {"overlap": -1}, {"overlap": 800}])
def test_chunkText_invalidParams_raise(bad: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        chunk_text("some text", **bad)


def test_parseDocument_textFile(tmp_path) -> None:  # type: ignore[no-untyped-def]
    f = tmp_path / "note.txt"
    f.write_text("plain text content", encoding="utf-8")
    assert parse_document(f) == "plain text content"


def test_parseDocument_missingFile_raises(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(FileNotFoundError):
        parse_document(tmp_path / "nope.txt")


def test_parseDocument_unsupportedType_raises(tmp_path) -> None:  # type: ignore[no-untyped-def]
    f = tmp_path / "image.png"
    f.write_bytes(b"\x89PNG")
    with pytest.raises(UnsupportedDocumentError):
        parse_document(f)


# --- service (both stores) -------------------------------------------------


def _fresh_chroma_store(embedder: FakeEmbeddings) -> ChromaVectorStore:
    # Unique collection name per store: chromadb.EphemeralClient() shares its
    # underlying storage across instances within a process (keyed by
    # collection name), so tests need distinct names to stay isolated.
    return ChromaVectorStore(
        chromadb.EphemeralClient(), embedder, collection_name=f"test-{uuid.uuid4().hex}"
    )


@pytest.fixture
def service(
    db: sqlite3.Connection, fake_embedder: FakeEmbeddings
) -> tuple[IngestionService, ChromaVectorStore, ChunkRepository]:
    UserRepository(db).create("A", user_id="u1")
    chunks = ChunkRepository(db)
    vectors = _fresh_chroma_store(fake_embedder)
    svc = IngestionService(
        documents=DocumentRepository(db),
        chunks=chunks,
        vector_store=vectors,
        candidate_profiles=CandidateProfileRepository(db),
    )
    return svc, vectors, chunks


def test_ingestText_writesToBothStores(
    service: tuple[IngestionService, ChromaVectorStore, ChunkRepository],
) -> None:
    svc, vectors, chunks = service
    doc_id = svc.ingest_text(
        user_id="u1",
        filename="resume.txt",
        text="Built Python services on AWS.\nLed a team of five engineers.",
        document_type=DocumentType.RESUME,
    )

    assert doc_id
    sqlite_rows = chunks.list_for_user("u1")
    assert len(sqlite_rows) >= 1
    # Same chunks are retrievable from Chroma, scoped to the user.
    found = vectors.search(user_id="u1", query="Python services on AWS", limit=10)
    assert len(found.chunks) == len(sqlite_rows)
    assert found.chunks[0].document_type is DocumentType.RESUME


def test_ingestFile_readsAndIngests(
    service: tuple[IngestionService, ChromaVectorStore, ChunkRepository],
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    svc, _vectors, chunks = service
    f = tmp_path / "cv.md"
    f.write_text("# Experience\nShipped a RAG system.", encoding="utf-8")

    svc.ingest_file(user_id="u1", path=str(f), document_type=DocumentType.RESUME)
    assert len(chunks.list_for_user("u1")) >= 1


def test_ingest_embeddingFailure_rollsBackSqlite(
    db: sqlite3.Connection, fake_embedder: FakeEmbeddings
) -> None:
    UserRepository(db).create("A", user_id="u1")
    chunks = ChunkRepository(db)

    class BoomStore(ChromaVectorStore):
        def upsert_chunks(self, chunks_arg):  # type: ignore[no-untyped-def]
            raise RuntimeError("chroma down")

    svc = IngestionService(
        documents=DocumentRepository(db),
        chunks=chunks,
        vector_store=BoomStore(
            chromadb.EphemeralClient(), fake_embedder, collection_name=f"test-{uuid.uuid4().hex}"
        ),
        candidate_profiles=CandidateProfileRepository(db),
    )

    with pytest.raises(RuntimeError):
        svc.ingest_text(user_id="u1", filename="r.txt", text="some content here")

    # SQLite chunk rows were rolled back so the stores don't drift.
    assert chunks.list_for_user("u1") == []


def test_deleteDocument_removesFromBothStores(
    service: tuple[IngestionService, ChromaVectorStore, ChunkRepository],
) -> None:
    svc, vectors, chunks = service
    doc_id = svc.ingest_text(user_id="u1", filename="r.txt", text="Python and AWS work.")
    assert chunks.list_for_user("u1")

    svc.delete_document(user_id="u1", document_id=doc_id)
    assert chunks.list_for_user("u1") == []
    assert vectors.search(user_id="u1", query="Python", limit=10).chunks == []


# --- candidate-profile cache invalidation -----------------------------------


def test_ingestText_flagsExistingCandidateProfileStale(
    db: sqlite3.Connection,
    service: tuple[IngestionService, ChromaVectorStore, ChunkRepository],
) -> None:
    from trustresume.models import CandidateProfile

    svc, _vectors, _chunks = service
    profiles = CandidateProfileRepository(db)
    profiles.upsert(user_id="u1", profile=CandidateProfile(name="A"), doc_hash="h1")
    assert profiles.get("u1").stale is False  # type: ignore[union-attr]

    svc.ingest_text(user_id="u1", filename="r2.txt", text="More AWS experience.")

    assert profiles.get("u1").stale is True  # type: ignore[union-attr]


def test_deleteDocument_flagsExistingCandidateProfileStale(
    db: sqlite3.Connection,
    service: tuple[IngestionService, ChromaVectorStore, ChunkRepository],
) -> None:
    from trustresume.models import CandidateProfile

    svc, _vectors, chunks = service
    doc_id = svc.ingest_text(user_id="u1", filename="r.txt", text="Python and AWS work.")
    profiles = CandidateProfileRepository(db)
    profiles.upsert(user_id="u1", profile=CandidateProfile(name="A"), doc_hash="h1")

    svc.delete_document(user_id="u1", document_id=doc_id)

    assert profiles.get("u1").stale is True  # type: ignore[union-attr]
    assert chunks.list_for_user("u1") == []


def test_ingestText_noCandidateProfileYet_doesNotError(
    service: tuple[IngestionService, ChromaVectorStore, ChunkRepository],
) -> None:
    """mark_stale is a no-op when there's no cache row yet — nothing to flag."""
    svc, _vectors, _chunks = service
    svc.ingest_text(user_id="u1", filename="r.txt", text="Some content.")  # must not raise
