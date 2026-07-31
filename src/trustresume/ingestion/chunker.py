"""Text cleaning and chunking.

Both functions are pure (text in, text out) so they're trivially testable and
carry no storage or model dependencies.

Chunking delegates to LangChain's ``RecursiveCharacterTextSplitter`` — the
framework's recommended general-purpose splitter — rather than a hand-rolled
loop. It splits recursively on the most semantic separator that fits (here:
paragraph boundary → space → character), which keeps a bullet or STAR story
intact where possible — the semantic unit the Trust Harness later retrieves
against — and hard-splits only when a single paragraph overflows ``max_chars``.
``chunk_overlap`` carries context across every boundary so a sentence
straddling a split still appears whole in an adjacent chunk.

``clean_text`` stays hand-rolled: it's whitespace normalization (a
domain-specific preprocessing step), not chunking, and it collapses blank
lines to a single ``\\n`` so the splitter's paragraph separator is exactly
``"\\n"`` (see ``_SEPARATORS``).

Milestone M3 (ingestion).
"""

from __future__ import annotations

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from trustresume.models import DocumentType, EvidenceChunk

# Collapses runs of blank lines into a paragraph break, and runs of intra-line
# whitespace into single spaces.
_MULTI_BLANK = re.compile(r"\n\s*\n+")
_INLINE_WS = re.compile(r"[ \t]+")

# ``clean_text`` collapses paragraph breaks to a single ``\n`` (not ``\n\n``),
# so the splitter's coarsest separator is ``"\n"``. The trailing ``""`` lets it
# fall back to a raw character split for a paragraph that overflows on its own.
_SEPARATORS = ["\n", " ", ""]


def clean_text(text: str) -> str:
    """Normalize whitespace without discarding paragraph structure.

    Paragraph breaks (blank lines) are preserved as single ``\\n`` separators
    because the chunker splits on them; other whitespace is collapsed.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_BLANK.sub("\n", text)
    lines = [_INLINE_WS.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def chunk_text(text: str, *, max_chars: int = 800, overlap: int = 100) -> list[str]:
    """Split cleaned text into chunks via ``RecursiveCharacterTextSplitter``.

    Recursively splits on the most semantic separator that keeps each chunk
    within ``max_chars`` (paragraph → space → character), carrying ``overlap``
    characters between adjacent chunks. Returns an empty list for empty input.
    """
    if not text.strip():
        return []
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must be >= 0 and < max_chars")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap,
        separators=_SEPARATORS,
        keep_separator=False,
    )
    return splitter.split_text(text)


def chunk_document(
    text: str,
    *,
    user_id: str,
    document_id: str,
    document_type: DocumentType = DocumentType.OTHER,
    source_document: str | None = None,
    max_chars: int = 800,
    overlap: int = 100,
) -> list[EvidenceChunk]:
    """``clean_text`` + ``chunk_text`` a document, wrapped into ``EvidenceChunk``s.

    Not currently called by :class:`~trustresume.ingestion.service.IngestionService`
    — it inlines ``clean_text``/``chunk_text`` directly and assigns each chunk a
    random id via its injectable ``id_factory``, so ``upsert_chunks`` retries
    don't collide with a prior partial write. This function exists for callers
    that want the "raw document text -> tagged, storable chunks" step as one
    call with *deterministic* ids (``{document_id}-{index}``, reproducible
    across repeated chunkings of the same document) — e.g. a retrieval-quality
    evaluation sweeping ``max_chars`` against a fixed document.
    """
    cleaned = clean_text(text)
    pieces = chunk_text(cleaned, max_chars=max_chars, overlap=overlap)
    return [
        EvidenceChunk(
            chunk_id=f"{document_id}-{index}",
            user_id=user_id,
            document_id=document_id,
            document_type=document_type,
            source_document=source_document,
            text=piece,
        )
        for index, piece in enumerate(pieces)
    ]
