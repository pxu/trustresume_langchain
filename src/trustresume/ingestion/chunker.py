"""Text cleaning and chunking.

Both functions are pure (text in, text out) so they're trivially testable and
carry no storage or model dependencies. The chunker splits on paragraph
boundaries first — career documents are paragraph/bullet structured, and
keeping a bullet or STAR story intact preserves the semantic unit the Trust
Harness later retrieves against. Oversized paragraphs are hard-split as a
fallback so no single chunk blows past the embedder's useful context.

Milestone M3 (ingestion).
"""

from __future__ import annotations

import re

from trustresume.models import DocumentType, EvidenceChunk

# Collapses runs of blank lines into a paragraph break, and runs of intra-line
# whitespace into single spaces.
_MULTI_BLANK = re.compile(r"\n\s*\n+")
_INLINE_WS = re.compile(r"[ \t]+")


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
    """Split cleaned text into chunks, preferring paragraph boundaries.

    Paragraphs are accumulated into a chunk until adding the next would exceed
    ``max_chars``; a paragraph longer than ``max_chars`` on its own is
    hard-split with ``overlap`` characters carried between pieces so a sentence
    straddling the boundary still appears whole in one chunk. Returns an empty
    list for empty input.
    """
    if not text.strip():
        return []
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must be >= 0 and < max_chars")

    paragraphs = [p for p in text.split("\n") if p]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > max_chars:
            # Flush what we have, then hard-split the oversized paragraph.
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_split(para, max_chars=max_chars, overlap=overlap))
            continue

        candidate = f"{current}\n{para}" if current else para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = para

    if current:
        chunks.append(current)
    return chunks


def _hard_split(text: str, *, max_chars: int, overlap: int) -> list[str]:
    """Split a single long string into overlapping windows of ``max_chars``."""
    step = max_chars - overlap
    pieces: list[str] = []
    start = 0
    while start < len(text):
        pieces.append(text[start : start + max_chars])
        start += step
    return pieces


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

    The reusable "raw document text -> tagged, storable chunks" step shared
    by :class:`~trustresume.ingestion.service.IngestionService` (fixed at the
    chunker's default ``max_chars``/``overlap``) and any caller that wants to
    experiment with different chunking parameters against the same document
    (e.g. a retrieval-quality evaluation sweeping ``max_chars``) without
    duplicating the clean -> chunk -> wrap sequence. Each chunk's ``chunk_id``
    is deterministic (``{document_id}-{index}``) rather than random, so
    re-chunking the same document at a different ``max_chars`` is
    reproducible.
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
