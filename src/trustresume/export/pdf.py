"""Render a resume draft as a simple, one-column PDF via ``fpdf2``.

Uses fpdf2's built-in Helvetica core font, which can only encode Latin-1. That
is a real constraint with a sharp edge: an LLM writing ordinary English prose
emits typographic punctuation constantly — em dashes, curly quotes, ellipses —
and *every one of those is outside Latin-1*. Left alone, fpdf2 raises, and
because ``TrustResumeApp._persist`` renders the PDF inline, that exception
would destroy a whole generation **after** all the LLM calls were already paid
for. A punctuation character is not worth losing a run over.

So text is normalized before rendering (:func:`_to_latin1`): typographic
punctuation is transliterated to its ASCII equivalent, and anything still
unencodable degrades to ``?`` with a warning rather than raising. The Markdown
export (``export/markdown.py``) is lossless and stays the faithful copy.

The remaining honest gap is scripts with no ASCII equivalent — a CJK or
Cyrillic name renders as ``?``. Fixing *that* properly means bundling a
Unicode TTF and calling ``FPDF.add_font``; it is a font-licensing and
repo-size decision, not a code one, and no current requirement forces it.
"""

from __future__ import annotations

import logging

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from trustresume.models import ResumeDraft

logger = logging.getLogger(__name__)

_MARGIN_MM = 15.0

#: Characters an LLM routinely produces that Latin-1 cannot encode, mapped to
#: the ASCII the writer almost certainly meant. Transliterating beats dropping
#: (an em dash silently vanishing changes how a sentence reads) and beats
#: raising (see the module docstring).
_TRANSLITERATIONS = str.maketrans(
    {
        "—": "--",  # em dash
        "–": "-",  # en dash
        "‒": "-",  # figure dash
        "‑": "-",  # non-breaking hyphen
        "−": "-",  # minus sign
        "‘": "'",  # left single quote
        "’": "'",  # right single quote / apostrophe
        "‚": ",",  # single low quote
        "“": '"',  # left double quote
        "”": '"',  # right double quote
        "„": '"',  # double low quote
        "…": "...",  # ellipsis
        "•": "-",  # bullet
        " ": " ",  # non-breaking space
        " ": " ",  # thin space
        " ": " ",  # narrow no-break space
        "­": "",  # soft hyphen — invisible; dropping loses nothing
        "​": "",  # zero-width space
        "﻿": "",  # BOM
        "™": "(TM)",
        "®": "(R)",
    }
)

#: Stand-in for a character with no ASCII equivalent (CJK, Cyrillic, emoji).
_UNSUPPORTED = "?"


def _to_latin1(text: str) -> str:
    """Text the core Helvetica font can actually encode.

    Two stages, deliberately in this order: transliterate what has a faithful
    ASCII equivalent, *then* replace whatever is left. Doing only the second
    would turn every em dash into ``?``, which is legible garbage; doing only
    the first would still raise on a CJK name.
    """
    transliterated = text.translate(_TRANSLITERATIONS)
    encodable = transliterated.encode("latin-1", errors="replace").decode("latin-1")
    if encodable != transliterated:
        # Worth a log line: the PDF a user downloads is now missing characters
        # that the Markdown export still has, and silent lossy output is how
        # someone ends up mailing a résumé with "?" where their name should be.
        logger.warning(
            "pdf export replaced characters unsupported by the core font",
            extra={"replacements": sum(c == _UNSUPPORTED for c in encodable)},
        )
    return encodable.replace("�", _UNSUPPORTED)


def _line(pdf: FPDF, w: float, h: float, text: str) -> None:
    """``multi_cell`` with the cursor reset to the left margin afterward.

    ``multi_cell``'s default ``new_x=XPos.RIGHT`` leaves the cursor at the
    *right* margin (verified empirically) — the next ``multi_cell(0, ...)``
    call (width ``0`` means "to the right margin from the current x") then
    computes a zero/negative width and raises ``FPDFException: Not enough
    horizontal space to render a single character``. Resetting to the left
    margin after every line avoids that for any number of consecutive lines.
    """
    pdf.multi_cell(w, h, text=_to_latin1(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


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
