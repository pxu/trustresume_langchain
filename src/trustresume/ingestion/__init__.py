"""Document ingestion: parse, clean, chunk, tag, embed, and store candidate documents.

Turns an uploaded career document into user-scoped, embedded evidence chunks
living in both stores (ADR-0001). Pure steps — ``parse_document``,
``clean_text``, ``chunk_text`` — are composed by :class:`IngestionService`,
which owns the store-syncing side effects.

Milestone M3 (ingestion).
"""

from __future__ import annotations

from .chunker import chunk_text, clean_text
from .parser import UnsupportedDocumentError, parse_document
from .service import IngestionService

__all__ = [
    "parse_document",
    "UnsupportedDocumentError",
    "clean_text",
    "chunk_text",
    "IngestionService",
]
