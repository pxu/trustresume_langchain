"""Rendering a generated resume draft into shareable export formats.

Pure functions, no framework dependency beyond ``fpdf2`` for the PDF
renderer — same "pure functions over a shared model" pattern as
``evaluation/``/``trust_verification/``. Every generated resume is rendered
to both forms and persisted alongside its scores (see
``api/app_service.py``'s ``_persist``), not generated on demand at download
time.
"""

from __future__ import annotations

from .markdown import render_markdown
from .pdf import render_pdf

__all__ = ["render_markdown", "render_pdf"]
