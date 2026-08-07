"""Unit tests for ``trustresume.export``: ``render_markdown``/``render_pdf``.

Pure functions, no SQLite/Chroma/LLM dependency — every branch (empty draft,
summary-only, sections-only, multiple sections, no bullets) is exercised
directly rather than only incidentally through ``_persist``.
"""

from __future__ import annotations

import logging

import pytest

from trustresume.export import render_markdown, render_pdf
from trustresume.export.pdf import _to_latin1
from trustresume.models import ResumeDraft, ResumeSection


def test_renderMarkdown_summaryAndSections_rendersBothInOrder() -> None:
    draft = ResumeDraft(
        summary="Experienced Python engineer.",
        sections=[
            ResumeSection(heading="Skills", bullets=["Python", "AWS"]),
            ResumeSection(heading="Experience", bullets=["Built RAG systems"]),
        ],
    )

    md = render_markdown(draft)

    assert md.startswith("Experienced Python engineer.")
    assert "## Skills" in md
    assert "- Python" in md
    assert "- AWS" in md
    assert "## Experience" in md
    assert "- Built RAG systems" in md
    # Sections come after the summary, in draft order.
    assert md.index("## Skills") < md.index("## Experience")


def test_renderMarkdown_noSummary_startsDirectlyWithFirstSectionHeading() -> None:
    draft = ResumeDraft(summary="", sections=[ResumeSection(heading="Skills", bullets=["Python"])])

    md = render_markdown(draft)

    assert md.startswith("## Skills")


def test_renderMarkdown_summaryOnly_noSections() -> None:
    draft = ResumeDraft(summary="Just a summary.", sections=[])

    assert render_markdown(draft) == "Just a summary."


def test_renderMarkdown_emptyDraft_returnsEmptyString() -> None:
    draft = ResumeDraft(summary="", sections=[])

    assert render_markdown(draft) == ""


def test_renderMarkdown_sectionWithNoBullets_stillRendersHeading() -> None:
    draft = ResumeDraft(summary="s", sections=[ResumeSection(heading="Empty", bullets=[])])

    md = render_markdown(draft)

    assert "## Empty" in md


def test_renderPdf_returnsValidPdfBytes() -> None:
    draft = ResumeDraft(
        summary="Experienced Python engineer.",
        sections=[ResumeSection(heading="Skills", bullets=["Python", "AWS"])],
    )

    pdf_bytes = render_pdf(draft)

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 100


def test_renderPdf_noSummary_stillRendersValidPdf() -> None:
    draft = ResumeDraft(summary="", sections=[ResumeSection(heading="Skills", bullets=["Python"])])

    assert render_pdf(draft).startswith(b"%PDF-")


def test_renderPdf_emptyDraft_stillRendersValidPdf() -> None:
    draft = ResumeDraft(summary="", sections=[])

    assert render_pdf(draft).startswith(b"%PDF-")


def test_renderPdf_sectionWithNoBullets_stillRendersValidPdf() -> None:
    draft = ResumeDraft(summary="s", sections=[ResumeSection(heading="Empty", bullets=[])])

    assert render_pdf(draft).startswith(b"%PDF-")


def test_renderPdf_manyBulletsOverflowingOnePage_stillRendersValidPdf() -> None:
    """A long draft (multi-page PDF) must render without error — exercises
    fpdf2's automatic page-break handling, not just a single short page.
    """
    bullets = [f"Bullet number {i} describing a specific achievement" for i in range(40)]
    draft = ResumeDraft(
        summary="s", sections=[ResumeSection(heading="Experience", bullets=bullets)]
    )

    pdf_bytes = render_pdf(draft)

    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 1000


# --- PDF text encoding -----------------------------------------------------
#
# The core Helvetica font only encodes Latin-1, and render_pdf runs inside
# _persist — so an unencodable character used to raise and destroy an entire
# generation *after* every LLM call had been paid for. A real Bedrock run hit
# this on an em dash, which models emit constantly.


@pytest.mark.parametrize(
    ("original", "expected_in_pdf_text"),
    [
        ("Senior Engineer — Platform", "Senior Engineer -- Platform"),
        ("Built the “order pipeline”", 'Built the "order pipeline"'),
        ("Python, Go, Terraform…", "Python, Go, Terraform..."),
        ("Led a team – of five", "Led a team - of five"),
        ("Owner’s manual", "Owner's manual"),
    ],
)
def test_toLatin1_typographicPunctuation_transliteratedNotDropped(
    original: str, expected_in_pdf_text: str
) -> None:
    """An em dash silently vanishing changes how a sentence reads."""
    assert _to_latin1(original) == expected_in_pdf_text


def test_renderPdf_llmTypographicPunctuation_rendersInsteadOfRaising() -> None:
    draft = ResumeDraft(
        summary="Backend engineer — 10 years’ experience…",
        sections=[
            ResumeSection(
                heading="Experience — Platform",
                bullets=["Owned the “order pipeline”", "• Led a team of five"],
            )
        ],
    )

    pdf_bytes = render_pdf(draft)

    assert pdf_bytes.startswith(b"%PDF-")


def test_renderPdf_charactersWithNoAsciiEquivalent_degradeRatherThanRaise() -> None:
    """A CJK name can't be transliterated, but it must not lose the whole run."""
    draft = ResumeDraft(summary="徐鹏飞 — 工程师 🚀", sections=[])

    assert render_pdf(draft).startswith(b"%PDF-")
    assert _to_latin1("徐鹏飞") == "???"


def test_renderPdf_lossyReplacement_isLogged(caplog: pytest.LogCaptureFixture) -> None:
    """Silent lossy output is how someone mails a résumé with '?' for a name."""
    with caplog.at_level(logging.WARNING, logger="trustresume.export.pdf"):
        render_pdf(ResumeDraft(summary="徐鹏飞", sections=[]))

    assert any("unsupported by the core font" in r.message for r in caplog.records)


def test_renderPdf_plainAsciiDraft_logsNothing(caplog: pytest.LogCaptureFixture) -> None:
    """The common case must stay quiet, or the warning becomes noise."""
    with caplog.at_level(logging.WARNING, logger="trustresume.export.pdf"):
        render_pdf(ResumeDraft(summary="Plain ASCII summary.", sections=[]))

    assert caplog.records == []


def test_renderMarkdown_keepsOriginalCharacters_unlikePdf() -> None:
    """Markdown is the lossless export; the PDF is best-effort under a core font."""
    draft = ResumeDraft(summary="Engineer — 徐鹏飞", sections=[])

    assert "—" in render_markdown(draft)
    assert "徐鹏飞" in render_markdown(draft)
