"""Unit tests for the ingestion pipeline.

Covers the pure steps (clean/chunk/parse) and the IngestionService's core
contract: chunks land in *both* stores, the stores stay in sync when embedding
fails, and delete removes from both.
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import chromadb
import pytest

from tests.fakes import FakeEmbeddings
from trustresume.ingestion import (
    IngestionService,
    UnsupportedDocumentError,
    chunk_text,
    clean_text,
    parse_bytes,
    parse_document,
)
from trustresume.ingestion.chunker import chunk_document
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


def test_chunkDocument_wrapsCleanedChunksAsEvidenceChunksWithDeterministicIds() -> None:
    chunks = chunk_document(
        "First paragraph.\n\nSecond paragraph.",
        user_id="u1",
        document_id="doc1",
        document_type=DocumentType.RESUME,
        source_document="r.txt",
    )
    assert [c.chunk_id for c in chunks] == [f"doc1-{i}" for i in range(len(chunks))]
    assert all(c.user_id == "u1" and c.document_id == "doc1" for c in chunks)
    assert all(c.document_type is DocumentType.RESUME for c in chunks)
    assert all(c.source_document == "r.txt" for c in chunks)


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


def test_parseBytes_textFile() -> None:
    assert parse_bytes("note.txt", b"plain text content") == "plain text content"


def test_parseBytes_unsupportedType_raises() -> None:
    with pytest.raises(UnsupportedDocumentError):
        parse_bytes("image.png", b"\x89PNG")


def test_parseBytes_textFile_invalidUtf8_raisesUnsupportedDocumentError() -> None:
    # A lone 0xff byte isn't valid UTF-8 (e.g. a Windows-1252 upload) — should
    # surface as the same UnsupportedDocumentError other unparseable uploads
    # get, not an uncaught UnicodeDecodeError.
    with pytest.raises(UnsupportedDocumentError):
        parse_bytes("note.txt", b"\xff\xfe bad encoding")


def test_parseBytes_richDocument_partitionRaises_wrappedAsUnsupportedDocumentError() -> None:
    # A corrupt/truncated .docx/.pdf can make `unstructured` raise almost
    # anything (zip errors, PDF parse errors, ...) — none of which are
    # UnsupportedDocumentError on their own, so this should be wrapped rather
    # than propagate as a raw exception.
    with (
        patch("unstructured.partition.auto.partition", side_effect=ValueError("bad zip")),
        pytest.raises(UnsupportedDocumentError),
    ):
        parse_bytes("cv.docx", b"not actually a docx")


def test_parseBytes_richDocument_joinsPartitionedElements() -> None:
    # .docx and .pdf share one code path through unstructured.partition —
    # unstructured.partition.auto.partition is mocked here to test the
    # element-joining logic without paying its (real, non-trivial) parse
    # cost; the real library is exercised end to end by
    # test_parseDocx_realSampleFile / test_parsePdf_realSampleFile below.
    fake_elements = [MagicMock(**{"__str__.return_value": "Experience"})]
    fake_elements.append(MagicMock(**{"__str__.return_value": "Shipped a RAG system."}))
    target = "unstructured.partition.auto.partition"
    with patch(target, return_value=fake_elements) as mock_partition:
        assert parse_bytes("cv.docx", b"fake docx bytes") == "Experience\nShipped a RAG system."
        assert mock_partition.call_args.kwargs["metadata_filename"] == "cv.docx"
        assert mock_partition.call_args.kwargs["strategy"] == "fast"


def test_parseBytes_richDocument_noElements_returnsEmptyString() -> None:
    with patch("unstructured.partition.auto.partition", return_value=[]):
        assert parse_bytes("cv.pdf", b"fake pdf bytes") == ""


@pytest.mark.live
def test_parseDocx_realSampleFile() -> None:
    """Exercises the real `unstructured` docx parser against a file on disk —
    live-marked because a first call loads real (non-trivial) parsing models,
    the same reason as the other `live` tests in this repo.
    """
    sample = Path(__file__).resolve().parents[2] / "data/sample_documents/AI_Engineer_Resume.docx"
    if not sample.exists():
        pytest.skip("sample docx not present in this checkout")
    text = parse_document(sample)
    assert "Peng Fei Xu" in text
    assert len(text) > 100


@pytest.mark.live
def test_parsePdf_realSampleFile() -> None:
    """Exercises the real `unstructured` PDF parser against a file on disk —
    live-marked for the same reason as test_parseDocx_realSampleFile above.
    """
    sample = Path(__file__).resolve().parents[2] / "data/sample_documents/Senior_SDE_Resume.pdf"
    if not sample.exists():
        pytest.skip("sample PDF not present in this checkout")
    text = parse_document(sample)
    assert "Peng Fei Xu" in text
    assert len(text) > 100


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


def test_ingestText_sameContentTwice_isNoOpAndReturnsSameDocumentId(
    service: tuple[IngestionService, ChromaVectorStore, ChunkRepository],
) -> None:
    """Re-ingesting the same resume content must not double every chunk in
    both stores — a duplicate upload (e.g. clicking "upload" twice, or
    re-uploading a resume that wasn't tracked as already ingested) is a
    no-op, not a second copy.
    """
    svc, vectors, chunks = service
    text = "Built Python services on AWS.\nLed a team of five engineers."

    first_id = svc.ingest_text(user_id="u1", filename="resume.txt", text=text)
    second_id = svc.ingest_text(user_id="u1", filename="resume.txt", text=text)

    assert first_id == second_id
    # Exactly the chunks from the FIRST ingest — nothing doubled in SQLite...
    sqlite_rows = chunks.list_for_user("u1")
    first_chunk_count = len(sqlite_rows)
    assert first_chunk_count >= 1
    # ...or in Chroma.
    found = vectors.search(user_id="u1", query="Python services on AWS", limit=10)
    assert len(found.chunks) == first_chunk_count


def test_ingestText_sameContentDifferentWhitespace_stillDeduped(
    service: tuple[IngestionService, ChromaVectorStore, ChunkRepository],
) -> None:
    """Content-hash dedup is computed on *cleaned* text, so two uploads that
    differ only in whitespace/line endings (e.g. the same resume re-exported
    from Word vs. Google Docs) still count as the same document.
    """
    svc, _vectors, chunks = service
    first_id = svc.ingest_text(user_id="u1", filename="v1.txt", text="Built Python   services.")
    second_id = svc.ingest_text(
        user_id="u1", filename="v2.txt", text="Built Python services.\r\n\r\n\r\n"
    )

    assert first_id == second_id
    assert len(chunks.list_for_user("u1")) == 1


def test_ingestText_sameContentDifferentUser_notDeduped(
    db: sqlite3.Connection, fake_embedder: FakeEmbeddings
) -> None:
    """Dedup is scoped by user_id — two different users uploading identical
    content (e.g. the same public résumé template) get separate documents,
    per the ADR-0001 user-isolation boundary.
    """
    UserRepository(db).create("A", user_id="u1")
    UserRepository(db).create("B", user_id="u2")
    vectors = _fresh_chroma_store(fake_embedder)
    svc = IngestionService(
        documents=DocumentRepository(db),
        chunks=ChunkRepository(db),
        vector_store=vectors,
        candidate_profiles=CandidateProfileRepository(db),
    )
    text = "Built Python services on AWS."

    id_u1 = svc.ingest_text(user_id="u1", filename="r.txt", text=text)
    id_u2 = svc.ingest_text(user_id="u2", filename="r.txt", text=text)

    assert id_u1 != id_u2


def test_ingestText_differentContent_bothIngested(
    service: tuple[IngestionService, ChromaVectorStore, ChunkRepository],
) -> None:
    """Sanity check on the other side of dedup: genuinely different content
    from the same user must NOT be treated as a duplicate.
    """
    svc, _vectors, chunks = service
    first_id = svc.ingest_text(user_id="u1", filename="resume.txt", text="Built Python services.")
    second_id = svc.ingest_text(
        user_id="u1", filename="cover_letter.txt", text="I am excited to apply."
    )

    assert first_id != second_id
    assert len(chunks.list_for_user("u1")) == 2


def test_ingestFile_readsAndIngests(
    service: tuple[IngestionService, ChromaVectorStore, ChunkRepository],
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    svc, _vectors, chunks = service
    f = tmp_path / "cv.md"
    f.write_text("# Experience\nShipped a RAG system.", encoding="utf-8")

    svc.ingest_file(user_id="u1", path=str(f), document_type=DocumentType.RESUME)
    assert len(chunks.list_for_user("u1")) >= 1


def test_ingestBytes_parsesAndIngestsInMemoryUpload(
    service: tuple[IngestionService, ChromaVectorStore, ChunkRepository],
) -> None:
    svc, vectors, chunks = service
    svc.ingest_bytes(
        user_id="u1",
        filename="resume.txt",
        data=b"Built Python services on AWS.",
        document_type=DocumentType.RESUME,
    )

    assert len(chunks.list_for_user("u1")) >= 1
    found = vectors.search(user_id="u1", query="Python services on AWS", limit=10)
    assert len(found.chunks) >= 1


def test_ingestBytes_unsupportedType_raisesBeforeWritingAnything(
    service: tuple[IngestionService, ChromaVectorStore, ChunkRepository],
) -> None:
    svc, _vectors, chunks = service
    with pytest.raises(UnsupportedDocumentError):
        svc.ingest_bytes(user_id="u1", filename="image.png", data=b"\x89PNG")
    assert chunks.list_for_user("u1") == []


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


def test_ingestText_concurrentDuplicateInsert_recoversViaExistingRow(
    db: sqlite3.Connection, fake_embedder: FakeEmbeddings
) -> None:
    """Regression test for the check-then-insert race: find_by_content_hash
    and DocumentRepository.create aren't one atomic transaction, so a
    concurrent duplicate ingest can slip between them and hit the
    idx_documents_user_content unique index instead. That must recover by
    re-reading the now-existing row, not propagate the raw IntegrityError.

    Simulated deterministically: patch find_by_content_hash to (falsely)
    report "not found" once, while a conflicting row already exists in the
    DB underneath it — reproducing the exact window a real race would open,
    without needing two actual concurrent threads.
    """
    from trustresume.ingestion.service import content_hash

    UserRepository(db).create("A", user_id="u1")
    documents = DocumentRepository(db)
    vectors = _fresh_chroma_store(fake_embedder)
    svc = IngestionService(
        documents=documents,
        chunks=ChunkRepository(db),
        vector_store=vectors,
        candidate_profiles=CandidateProfileRepository(db),
    )
    text = "Built Python services on AWS."
    winning_id = documents.create(
        user_id="u1",
        filename="raced-in-first.txt",
        document_type=DocumentType.OTHER,
        content_hash=content_hash(clean_text(text)),
    )

    text_hash = content_hash(clean_text(text))
    real_result = documents.find_by_content_hash(user_id="u1", content_hash=text_hash)

    # Lie ("not found") on the first call only — the fallback re-read inside
    # ingest_text's `except IntegrityError` branch must see the real row.
    with patch.object(
        documents, "find_by_content_hash", side_effect=[None, real_result]
    ) as mock_find:
        result_id = svc.ingest_text(user_id="u1", filename="resume.txt", text=text)
    assert mock_find.call_count == 2  # the lied-to check, then the recovery re-read

    assert result_id == winning_id


def test_ingestText_integrityErrorWithNoMatchingRow_raisesRuntimeError(
    db: sqlite3.Connection, fake_embedder: FakeEmbeddings
) -> None:
    """If the fallback re-read after an IntegrityError finds nothing (a
    scenario stranger than an ordinary race — e.g. the conflicting row was
    deleted between the failed insert and the re-read), ingest_text must
    raise a clear RuntimeError rather than crash on `existing["id"]` with a
    confusing TypeError, or (if this were still a bare assert) silently
    vanish under `python -O`.
    """
    UserRepository(db).create("A", user_id="u1")
    documents = DocumentRepository(db)
    svc = IngestionService(
        documents=documents,
        chunks=ChunkRepository(db),
        vector_store=_fresh_chroma_store(fake_embedder),
        candidate_profiles=CandidateProfileRepository(db),
    )
    text = "Built Python services on AWS."

    with (
        patch.object(documents, "find_by_content_hash", side_effect=[None, None]),
        patch.object(documents, "create", side_effect=sqlite3.IntegrityError("boom")),
        pytest.raises(RuntimeError, match="no matching document row"),
    ):
        svc.ingest_text(user_id="u1", filename="resume.txt", text=text)


def test_deleteDocument_removesFromBothStores(
    service: tuple[IngestionService, ChromaVectorStore, ChunkRepository],
) -> None:
    svc, vectors, chunks = service
    doc_id = svc.ingest_text(user_id="u1", filename="r.txt", text="Python and AWS work.")
    assert chunks.list_for_user("u1")

    svc.delete_document(user_id="u1", document_id=doc_id)
    assert chunks.list_for_user("u1") == []
    assert vectors.search(user_id="u1", query="Python", limit=10).chunks == []


def test_deleteDocument_thenReingestSameContent_actuallyReingests(
    service: tuple[IngestionService, ChromaVectorStore, ChunkRepository],
) -> None:
    """Regression test: deleting a document must also delete its `documents`
    row, not just its chunks — otherwise the deleted row's content_hash stays
    live in the dedup index forever, and re-ingesting identical content after
    a delete would match that now-empty, orphaned row and silently no-op
    (returning a "successful" document id with zero chunks) instead of
    actually re-ingesting.
    """
    svc, vectors, chunks = service
    text = "Python and AWS work."
    first_id = svc.ingest_text(user_id="u1", filename="r.txt", text=text)
    svc.delete_document(user_id="u1", document_id=first_id)
    assert chunks.list_for_user("u1") == []

    second_id = svc.ingest_text(user_id="u1", filename="r.txt", text=text)

    # A fresh document, not a no-op match against the deleted row's id.
    assert second_id != first_id
    assert len(chunks.list_for_user("u1")) >= 1
    assert vectors.search(user_id="u1", query="Python AWS", limit=10).chunks


def test_ingest_embeddingFailure_thenRetrySameContent_actuallyIngests(
    db: sqlite3.Connection, fake_embedder: FakeEmbeddings
) -> None:
    """Regression test: rolling back a failed Chroma upsert must also delete
    the `documents` row it just created, not only the chunk rows — otherwise
    that row's content_hash permanently blocks any future retry of the same
    content (find_by_content_hash would keep matching the orphaned,
    chunk-less row and silently no-op instead of actually retrying).
    """
    from trustresume.ingestion.service import content_hash

    UserRepository(db).create("A", user_id="u1")
    chunks = ChunkRepository(db)
    documents = DocumentRepository(db)

    class BoomOnce(ChromaVectorStore):
        _armed = True

        def upsert_chunks(self, chunks_arg):  # type: ignore[no-untyped-def]
            if type(self)._armed:
                raise RuntimeError("chroma down")
            super().upsert_chunks(chunks_arg)

    vectors = BoomOnce(
        chromadb.EphemeralClient(), fake_embedder, collection_name=f"test-{uuid.uuid4().hex}"
    )
    svc = IngestionService(
        documents=documents,
        chunks=chunks,
        vector_store=vectors,
        candidate_profiles=CandidateProfileRepository(db),
    )
    text = "some content here"

    with pytest.raises(RuntimeError):
        svc.ingest_text(user_id="u1", filename="r.txt", text=text)
    assert chunks.list_for_user("u1") == []

    BoomOnce._armed = False
    doc_id = svc.ingest_text(user_id="u1", filename="r.txt", text=text)

    assert len(chunks.list_for_user("u1")) >= 1
    assert vectors.search(user_id="u1", query="some content here", limit=10).chunks
    assert (
        documents.find_by_content_hash(user_id="u1", content_hash=content_hash(clean_text(text)))[
            "id"
        ]
        == doc_id
    )


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


def test_deleteDocument_vectorStoreFails_stillFlagsCandidateProfileStale(
    db: sqlite3.Connection,
    service: tuple[IngestionService, ChromaVectorStore, ChunkRepository],
) -> None:
    """A Chroma failure mid-delete must not skip mark_stale: the SQLite chunk
    rows are already gone by that point, so the cached Candidate Profile
    would otherwise keep describing documents that no longer exist.
    """
    from trustresume.models import CandidateProfile

    svc, vectors, _chunks = service
    doc_id = svc.ingest_text(user_id="u1", filename="r.txt", text="Python and AWS work.")
    profiles = CandidateProfileRepository(db)
    profiles.upsert(user_id="u1", profile=CandidateProfile(name="A"), doc_hash="h1")

    with (
        patch.object(vectors, "delete_chunks", side_effect=RuntimeError("chroma unavailable")),
        pytest.raises(RuntimeError, match="chroma unavailable"),
    ):
        svc.delete_document(user_id="u1", document_id=doc_id)

    assert profiles.get("u1").stale is True  # type: ignore[union-attr]


def test_ingestText_noCandidateProfileYet_doesNotError(
    service: tuple[IngestionService, ChromaVectorStore, ChunkRepository],
) -> None:
    """mark_stale is a no-op when there's no cache row yet — nothing to flag."""
    svc, _vectors, _chunks = service
    svc.ingest_text(user_id="u1", filename="r.txt", text="Some content.")  # must not raise
