"""Text extraction from uploaded documents.

Plain-text/markdown are read directly. Everything else (``.docx``, ``.pdf``,
and any other format `unstructured` recognizes) goes through one shared
entry point, `unstructured.partition.auto.partition` — the industry-standard
document-parsing library for RAG pipelines. One partition call replaces what
would otherwise be a per-format parsing function (python-docx for .docx,
pypdf for .pdf, ...), so adding another office-document format later is a
one-line addition to `_RICH_SUFFIXES`, not a new parsing function.

``strategy="fast"`` (pure text extraction, no layout/OCR model) — this
project only needs the text for chunking, not a layout-aware reader, and the
default ``"hi_res"`` strategy loads a torch-based layout model that costs
~10-20x the latency for no benefit here.

Milestone M3 (ingestion); PDF added post-M3; both migrated to `unstructured`
(replacing pypdf/python-docx as separate per-format branches) after that.
"""

from __future__ import annotations

import io
from pathlib import Path

_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".text"}
_RICH_SUFFIXES = {".docx", ".pdf"}


class UnsupportedDocumentError(ValueError):
    """Raised for a file type the parser can't extract text from."""


def parse_document(path: str | Path) -> str:
    """Extract raw text from a document on disk, by file extension.

    Raises :class:`UnsupportedDocumentError` for unknown types and
    ``FileNotFoundError`` for a missing file — ingestion should stop rather
    than store empty or malformed content.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return parse_bytes(p.name, p.read_bytes())


def parse_bytes(filename: str, data: bytes) -> str:
    """Extract raw text from an in-memory upload, by ``filename``'s extension.

    The in-memory counterpart to :func:`parse_document` — for callers that
    receive bytes directly (a web upload, ``UploadFile.read()``) and shouldn't
    have to write a temp file just to reuse the file-based parser. Same
    extension dispatch and errors as :func:`parse_document`.
    """
    suffix = Path(filename).suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UnsupportedDocumentError(
                f"Could not decode {filename!r} as UTF-8 text: {exc}"
            ) from exc
    if suffix in _RICH_SUFFIXES:
        return _parse_rich_bytes(filename, data)
    raise UnsupportedDocumentError(f"Unsupported document type: {suffix or '(no extension)'}")


def _parse_rich_bytes(filename: str, data: bytes) -> str:
    """Extract text from .docx/.pdf bytes via ``unstructured``, one element per line."""
    from unstructured.partition.auto import partition  # imported lazily; a heavy dependency

    try:
        elements = partition(file=io.BytesIO(data), metadata_filename=filename, strategy="fast")
    except Exception as exc:
        # `unstructured` can raise almost anything for a corrupt/truncated
        # upload (zip errors for .docx, PDF parse errors, ...) — none of
        # which are UnsupportedDocumentError, so callers that only catch that
        # (e.g. the API's upload route) would otherwise see a raw 500
        # instead of the same clean 422 other unparseable files get.
        raise UnsupportedDocumentError(f"Could not parse {filename!r}: {exc}") from exc
    return "\n".join(str(element) for element in elements)
