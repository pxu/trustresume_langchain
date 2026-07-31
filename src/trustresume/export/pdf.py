"""Render a resume draft as a simple, one-column PDF via ``fpdf2``.

Uses the built-in Helvetica core font (no bundled Unicode TTF), so
non-Latin candidate text isn't supported here — no current requirement
covers it; adding it later would mean bundling a Unicode font and calling
``FPDF.add_font`` instead of relying on the built-in core fonts.
"""

from __future__ import annotations

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from trustresume.models import ResumeDraft

_MARGIN_MM = 15.0


def _line(pdf: FPDF, w: float, h: float, text: str) -> None:
    """``multi_cell`` with the cursor reset to the left margin afterward.

    ``multi_cell``'s default ``new_x=XPos.RIGHT`` leaves the cursor at the
    *right* margin (verified empirically) — the next ``multi_cell(0, ...)``
    call (width ``0`` means "to the right margin from the current x") then
    computes a zero/negative width and raises ``FPDFException: Not enough
    horizontal space to render a single character``. Resetting to the left
    margin after every line avoids that for any number of consecutive lines.
    """
    pdf.multi_cell(w, h, text=text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def render_pdf(draft: ResumeDraft) -> bytes:
    """Render ``draft`` as PDF bytes: summary paragraph, then one heading +
    bullet list per section, in draft order.
    """
    pdf = FPDF()
    pdf.set_margin(_MARGIN_MM)
    pdf.add_page()

    if draft.summary:
        pdf.set_font("Helvetica", size=11)
        _line(pdf, 0, 6, draft.summary)
        pdf.ln(4)

    for section in draft.sections:
        pdf.set_font("Helvetica", "B", 13)
        _line(pdf, 0, 8, section.heading)
        pdf.set_font("Helvetica", size=11)
        for bullet in section.bullets:
            _line(pdf, 0, 6, f"- {bullet}")
        pdf.ln(2)

    # ``output()`` with no filename returns the document as a bytearray
    # (verified against fpdf2's own signature) rather than writing to disk.
    return bytes(pdf.output())
