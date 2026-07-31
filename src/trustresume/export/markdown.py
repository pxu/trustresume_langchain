"""Render a resume draft as plain Markdown.

Pure string formatting — no dependency beyond ``models``.
"""

from __future__ import annotations

from trustresume.models import ResumeDraft


def render_markdown(draft: ResumeDraft) -> str:
    """Render ``draft`` as GitHub-flavored Markdown.

    The summary (if any) as a plain paragraph, then one ``## {heading}``
    per section with its bullets as a ``- {bullet}`` list, in draft order.
    An entirely empty draft (no summary, no sections) renders as an empty
    string rather than raising — a caller persisting/downloading a draft
    that happens to be blank still gets a well-defined result.
    """
    lines: list[str] = []
    if draft.summary:
        lines.append(draft.summary)
    for section in draft.sections:
        if lines:
            lines.append("")
        lines.append(f"## {section.heading}")
        lines.extend(f"- {bullet}" for bullet in section.bullets)
    return "\n".join(lines)
