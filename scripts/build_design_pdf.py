"""Render docs/detailed-design.md (+ docs/diagrams/*.png) into docs/detailed-design.pdf.

Throwaway doc-tooling, like generate_design_diagrams.py — not part of the app,
not covered by the test suite. This environment has no network access and no
pandoc/wkhtmltopdf/weasyprint installed, so this hand-rolls a small Markdown
subset -> HTML converter (headings, paragraphs, bold/italic/inline-code,
fenced code blocks, bullet lists, and pipe tables — the only constructs
docs/detailed-design.md actually uses) and feeds the result to fpdf2's HTML writer,
which the app already depends on for résumé PDF export
(trustresume.export.pdf). Diagram images are spliced in at fixed anchor
points keyed to section headings.

Run:

    .venv/bin/python scripts/generate_design_diagrams.py   # regenerate PNGs first
    .venv/bin/python scripts/build_design_pdf.py
"""

from __future__ import annotations

import html as htmllib
import os
import re
from pathlib import Path

import matplotlib
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from fpdf.fonts import TextStyle

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MD = ROOT / "docs/detailed-design.md"
DIAGRAMS_DIR = ROOT / "docs" / "diagrams"
OUTPUT_PDF = ROOT / "docs/detailed-design.pdf"

_FONT_DIR = Path(matplotlib.__file__).resolve().parent / "mpl-data" / "fonts" / "ttf"

# Diagrams to splice in right after the H2 heading whose text contains the
# given substring (matched case-sensitively against the heading's own text,
# after stripping the "N. " numbering) — keeps the mapping readable without
# hardcoding line numbers that would drift if the doc is re-ordered.
_DIAGRAM_ANCHORS: list[tuple[str, str, str]] = [
    (
        "What the system does",
        "01_system_overview.png",
        "Figure 1. System overview — UI through HTTP to the facade, orchestrator, "
        "agents, and the two backing stores.",
    ),
    (
        "Orchestration (`orchestration/orchestrator.py`)",
        "02_orchestrator_graph.png",
        "Figure 2. The LangGraph StateGraph — one-time setup nodes, the repeatable "
        "quality loop, and the pre-increment iteration check that yields 4 drafts.",
    ),
    (
        "End-to-end data flow",
        "03_generation_sequence.png",
        "Figure 3. One POST /api/generate call, start to finish.",
    ),
    (
        "Storage schema (`storage/schema.py`, SQLite)",
        "04_storage_schema.png",
        "Figure 4. The SQLite schema — every table except users carries user_id.",
    ),
    (
        "Deployment",
        "05_deployment.png",
        "Figure 5. Docker Compose topology — one image, two services.",
    ),
]

_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_CODE_PLACEHOLDER = "\x00{}\x00"


def _extract_code_spans(text: str) -> tuple[str, list[str]]:
    """Pull every `` `...` `` span out of ``text``, leaving a numbered placeholder.

    Several code spans in this doc contain a bare ``*`` (e.g. `` `chunk_text(text,
    *, max_chars=800, ...)` ``) — applying the bold/italic regexes to the whole
    paragraph in one pass lets that stray ``*`` pair up with an unrelated ``*``
    elsewhere in the same paragraph, producing italic markup that straddles
    (and corrupts) the code span's own tag. Pulling code spans out first, then
    running bold/italic only on what's left, then splicing the (verbatim,
    already-escaped) code back in by placeholder, makes that impossible: the
    bold/italic regexes never see a code span's contents at all.
    """
    spans: list[str] = []

    def _stash(m: re.Match[str]) -> str:
        spans.append(m.group(1))
        return _CODE_PLACEHOLDER.format(len(spans) - 1)

    return _INLINE_CODE_RE.sub(_stash, text), spans


def _inline(text: str) -> str:
    """Markdown inline spans -> HTML, applied to already-escaped text is wrong
    order — so this escapes first, then re-introduces the few HTML tags this
    doc's inline markdown needs (code/b/i), never letting raw doc text pass
    through unescaped.
    """
    without_code, spans = _extract_code_spans(text)
    escaped = htmllib.escape(without_code, quote=False)
    escaped = _BOLD_RE.sub(lambda m: f"<b>{m.group(1)}</b>", escaped)
    escaped = _ITALIC_RE.sub(lambda m: f"<i>{m.group(1)}</i>", escaped)
    escaped = _LINK_RE.sub(lambda m: f"<u>{m.group(1)}</u>", escaped)  # no live links needed
    for i, code in enumerate(spans):
        placeholder = _CODE_PLACEHOLDER.format(i)
        replacement = f'<font face="DejaVuMono">{htmllib.escape(code, quote=False)}</font>'
        escaped = escaped.replace(placeholder, replacement)
    return escaped


def _inline_plain(text: str) -> str:
    """Table-cell variant of :func:`_inline`.

    fpdf2's HTML table cells (``handle_data``) accept only one flat text run
    per ``<td>``/``<th>`` — a second nested tag (our ``<font>``/``<b>``/``<i>``
    inline-code/bold/italic markup) raises ``NotImplementedError``. Every
    table in this doc uses inline code/bold liberally, so cells strip the
    Markdown markers down to plain text instead of converting them to tags.
    """
    without_code, spans = _extract_code_spans(text)
    plain = _BOLD_RE.sub(lambda m: m.group(1), without_code)
    plain = _ITALIC_RE.sub(lambda m: m.group(1), plain)
    plain = _LINK_RE.sub(lambda m: m.group(1), plain)
    for i, code in enumerate(spans):
        plain = plain.replace(_CODE_PLACEHOLDER.format(i), code)
    return htmllib.escape(plain, quote=False)


def _consume_list_items(lines: list[str], start: int, *, marker_re: str) -> tuple[list[str], int]:
    """Read a contiguous run of list items starting at ``lines[start]``.

    A wrapped item's continuation is an indented line with no marker of its
    own (this doc always wraps that way, never with a nested sub-list) — fold
    it into the current item's text rather than starting a new one. Returns
    ``(item_texts, lines_consumed)``.
    """
    marker = re.compile(marker_re)
    items: list[str] = []
    i = start
    while i < len(lines):
        stripped = lines[i].strip()
        m = marker.match(stripped)
        if m:
            items.append(stripped[m.end() :])
            i += 1
        elif stripped and lines[i].startswith((" ", "\t")) and items:
            items[-1] += " " + stripped
            i += 1
        else:
            break
    return items, i - start


def _find_anchor(heading_text: str, remaining: dict[str, tuple[str, str]]) -> str | None:
    """The first not-yet-used anchor key that's a substring of this heading's text."""
    for key in remaining:
        if key in heading_text:
            return key
    return None


def markdown_to_html(md_text: str) -> str:
    lines = md_text.splitlines()
    anchors_remaining = {k: (img, cap) for k, img, cap in _DIAGRAM_ANCHORS}

    out: list[str] = []
    i = 0
    in_table: list[list[str]] = []
    table_header: list[str] | None = None

    def flush_table() -> None:
        nonlocal in_table, table_header
        if not in_table:
            return
        out.append('<table border="1" cellpadding="4">')
        if table_header is not None:
            out.append(
                "<thead><tr>"
                + "".join(f"<th>{_inline_plain(c)}</th>" for c in table_header)
                + "</tr></thead>"
            )
        for row in in_table:
            out.append("<tr>" + "".join(f"<td>{_inline_plain(c)}</td>" for c in row) + "</tr>")
        out.append("</table>")
        in_table = []
        table_header = None

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Fenced code block
        if stripped.startswith("```"):
            flush_table()
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            code = htmllib.escape("\n".join(code_lines), quote=False)
            out.append(f'<pre><font face="DejaVuMono">{code}</font></pre>')
            continue

        # Table row
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if re.match(r"^:?-+:?$", cells[0].replace(" ", "")) or all(
                re.match(r"^:?-+:?$", c.replace(" ", "")) for c in cells
            ):
                # separator row (---|---) — signals the previous row was the header
                if out and not table_header and in_table:
                    table_header = in_table.pop()
                i += 1
                continue
            in_table.append(cells)
            i += 1
            continue
        else:
            flush_table()

        # Headings
        m = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if m:
            level = len(m.group(1))
            heading_text = m.group(2)
            tag = {1: "h1", 2: "h2", 3: "h3", 4: "h4"}[level]
            out.append(f"<{tag}>{_inline(heading_text)}</{tag}>")

            key = _find_anchor(heading_text, anchors_remaining)
            if key is not None:
                img_name, caption = anchors_remaining.pop(key)
                img_path = DIAGRAMS_DIR / img_name
                out.append(
                    f'<p align="center"><img src="{img_path.as_posix()}" width="620"></p>'
                )
                out.append(
                    f'<p align="center"><font face="DejaVu" size="9"><i>{_inline(caption)}</i></font></p>'
                )
            i += 1
            continue

        # No blockquote handling: this doc has no intentional `> ` blockquotes
        # — every occurrence is a hard-wrapped mid-paragraph line that happens
        # to start with `>` used as "implies" (e.g. "env var > llm.json >
        # default"), so treating a leading `> ` as a block construct would
        # misparse those. Falls through to the plain-paragraph case below.

        # Ordered list — consume a contiguous run, folding indented
        # continuation lines (this doc always wraps long items onto an
        # indented next line rather than one long line) into the same item.
        if re.match(r"^\d+\.\s+", stripped):
            items, consumed = _consume_list_items(lines, i, marker_re=r"^\d+\.\s+")
            i += consumed
            out.append("<ol>" + "".join(f"<li>{_inline(it)}</li>" for it in items) + "</ol>")
            continue

        # Unordered list — same continuation-folding as above.
        if stripped.startswith("- ") or stripped.startswith("* "):
            items, consumed = _consume_list_items(lines, i, marker_re=r"^[-*]\s+")
            i += consumed
            out.append("<ul>" + "".join(f"<li>{_inline(it)}</li>" for it in items) + "</ul>")
            continue

        # Horizontal rule
        if stripped == "---":
            out.append("<hr>")
            i += 1
            continue

        # Blank line
        if not stripped:
            i += 1
            continue

        # Plain paragraph — consume a contiguous run of non-blank, non-special lines
        para_lines = [stripped]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,4})\s", lines[i].strip()):
            nxt = lines[i].strip()
            if nxt.startswith(("```", "|", "- ", "* ")) or nxt == "---":
                break
            para_lines.append(nxt)
            i += 1
        out.append(f"<p>{_inline(' '.join(para_lines))}</p>")

    flush_table()

    leftover = list(anchors_remaining)
    if leftover:
        raise RuntimeError(f"diagram anchors never matched a heading: {leftover}")

    return "\n".join(out)


TAG_STYLE_CSS = """
h1 { color: #1f3552; }
h2 { color: #1f3552; }
h3 { color: #1f3552; }
"""


def build_pdf() -> None:
    md_text = SOURCE_MD.read_text(encoding="utf-8")
    body_html = markdown_to_html(md_text)

    pdf = FPDF(format="Letter")
    pdf.add_font("DejaVu", "", str(_FONT_DIR / "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", str(_FONT_DIR / "DejaVuSans-Bold.ttf"))
    pdf.add_font("DejaVu", "I", str(_FONT_DIR / "DejaVuSans-Oblique.ttf"))
    pdf.add_font("DejaVu", "BI", str(_FONT_DIR / "DejaVuSans-BoldOblique.ttf"))
    pdf.add_font("DejaVuMono", "", str(_FONT_DIR / "DejaVuSansMono.ttf"))
    pdf.add_font("DejaVuMono", "B", str(_FONT_DIR / "DejaVuSansMono-Bold.ttf"))
    pdf.add_font("DejaVuMono", "I", str(_FONT_DIR / "DejaVuSansMono-Oblique.ttf"))
    pdf.add_font("DejaVuMono", "BI", str(_FONT_DIR / "DejaVuSansMono-BoldOblique.ttf"))
    pdf.set_font("DejaVu", size=11)
    pdf.set_margin(18)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Title page content, then the converted body. Every multi_cell call below
    # resets the cursor to the left margin afterward (new_x=LMARGIN) — its
    # default (new_x=RIGHT) leaves the next multi_cell(0, ...) computing a
    # zero/negative width, the same FPDFException export/pdf.py's `_line`
    # helper works around for résumé export.
    pdf.set_font("DejaVu", "B", 26)
    pdf.ln(40)
    pdf.multi_cell(0, 14, "TrustResume", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("DejaVu", "", 16)
    pdf.multi_cell(0, 10, "Detailed Design", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)
    pdf.set_font("DejaVu", "I", 11)
    pdf.multi_cell(
        0,
        7,
        "A comprehensive reference for the codebase's architecture, data flow,\n"
        "testing model, and deployment — companion to docs/architecture/high-level-design.md\n"
        "and docs/code-walkthrough.md.",
        align="C",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.add_page()

    pdf.write_html(
        body_html,
        table_line_separators=True,
        tag_styles={
            "h1": TextStyle(font_family="DejaVu", font_style="B", font_size_pt=20, color="#1f3552"),
            "h2": TextStyle(font_family="DejaVu", font_style="B", font_size_pt=16, color="#1f3552"),
            "h3": TextStyle(font_family="DejaVu", font_style="B", font_size_pt=13, color="#1f3552"),
            "h4": TextStyle(font_family="DejaVu", font_style="B", font_size_pt=11.5, color="#1f3552"),
            "li": TextStyle(l_margin=6),
            "pre": TextStyle(font_family="DejaVuMono", font_size_pt=9),
            "code": TextStyle(font_family="DejaVuMono"),
        },
    )

    pdf.output(str(OUTPUT_PDF))
    print(f"wrote {OUTPUT_PDF} ({OUTPUT_PDF.stat().st_size / 1024:.0f} KiB, {pdf.page_no()} pages)")


if __name__ == "__main__":
    build_pdf()
