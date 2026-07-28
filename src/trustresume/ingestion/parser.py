"""Text extraction from uploaded documents.

Kept deliberately small: plain-text/markdown are read directly, ``.docx`` via
python-docx (already a declared dependency for resume export). Anything else
raises so ingestion fails loudly rather than embedding garbage. PDF is a
documented follow-up — the design targets user-authored career docs, which are
overwhelmingly .docx/.txt/.md.

Milestone M3 (ingestion).
"""

from __future__ import annotations

from pathlib import Path

_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".text"}


class UnsupportedDocumentError(ValueError):
    """Raised for a file type the parser can't extract text from."""


def parse_document(path: str | Path) -> str:
    """Extract raw text from a document by file extension.

    Raises :class:`UnsupportedDocumentError` for unknown types and
    ``FileNotFoundError`` for a missing file — ingestion should stop rather
    than store empty or malformed content.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    suffix = p.suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        return p.read_text(encoding="utf-8")
    if suffix == ".docx":
        return _parse_docx(p)
    raise UnsupportedDocumentError(f"Unsupported document type: {suffix or '(no extension)'}")


def _parse_docx(path: Path) -> str:
    """Extract paragraph text from a .docx file, one paragraph per line."""
    from docx import Document  # imported lazily; only needed for .docx

    document = Document(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)
