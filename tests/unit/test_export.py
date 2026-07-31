"""Unit tests for ``trustresume.export``: ``render_markdown``/``render_pdf``.

Pure functions, no SQLite/Chroma/LLM dependency — every branch (empty draft,
summary-only, sections-only, multiple sections, no bullets) is exercised
directly rather than only incidentally through ``_persist``.
"""

from __future__ import annotations

from trustresume.export import render_markdown, render_pdf
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
