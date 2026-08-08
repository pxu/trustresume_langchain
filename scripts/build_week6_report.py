"""Generate docs/weekly-submissions/Week6_Model_Testing_and_Debugging_Report.docx.

Throwaway doc-tooling, like build_design_pdf.py/generate_sample_documents.py —
not part of the app, not covered by the test suite. Mirrors the formatting of
the (hand-authored, in Word) Week 5 report so the two look like one series:
Times New Roman, double-spaced, centered title page, bold numbered section
headings, "List Bullet" for bullet lists.

    .venv/bin/python scripts/build_week6_report.py
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "weekly-submissions" / "Week6_Model_Testing_and_Debugging_Report.docx"

FONT_NAME = "Times New Roman"
FONT_SIZE = Pt(12)


def _set_run_font(run) -> None:  # type: ignore[no-untyped-def]
    run.font.name = FONT_NAME
    run.font.size = FONT_SIZE


def add_para(doc: Document, text: str = "", *, center: bool = False, bold: bool = False) -> None:  # type: ignore[no-untyped-def]
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 2.0
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if text:
        run = p.add_run(text)
        run.bold = bold
        _set_run_font(run)


def add_body(doc: Document, text: str) -> None:  # type: ignore[no-untyped-def]
    """A double-spaced body paragraph, left-aligned (Word's default justify-none)."""
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 2.0
    run = p.add_run(text)
    _set_run_font(run)


def add_heading(doc: Document, text: str) -> None:  # type: ignore[no-untyped-def]
    add_para(doc, text, bold=True)


def add_bullets(doc: Document, items: list[str]) -> None:  # type: ignore[no-untyped-def]
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.line_spacing = 2.0
        run = p.add_run(item)
        _set_run_font(run)


def add_table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:  # type: ignore[no-untyped-def]
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Light Grid Accent 1"
    for cell, text in zip(table.rows[0].cells, headers, strict=True):
        run = cell.paragraphs[0].add_run(text)
        run.bold = True
        _set_run_font(run)
    for row, values in zip(table.rows[1:], rows, strict=True):
        for cell, text in zip(row.cells, values, strict=True):
            run = cell.paragraphs[0].add_run(text)
            _set_run_font(run)
    doc.add_paragraph()


def build() -> Document:  # type: ignore[no-untyped-def]
    doc = Document()

    for style_name in ("Normal", "List Bullet"):
        style = doc.styles[style_name]
        style.font.name = FONT_NAME
        style.font.size = FONT_SIZE

    # --- Title page ---------------------------------------------------------
    add_para(doc, "", center=True)
    add_para(doc, "MSAI 699 Capstone Project", center=True)
    add_para(doc, "", center=True)
    add_para(doc, "Instructor: Dr. Gamini Bulumulle", center=True)
    add_para(doc, "", center=True)
    add_para(doc, "Week 6: Testing, Evaluation & Documentation: Model Testing and Debugging Report", center=True)
    add_para(doc, "", center=True)
    add_para(doc, "Peng Fei Xu", center=True)
    add_para(doc, "", center=True)
    add_para(doc, "University of the Cumberlands", center=True)
    add_para(doc, "", center=True)
    add_para(doc, "Submitted to the University of the Cumberlands", center=True)
    add_para(doc, "in Partial Fulfillment of the Requirements of the Degree of", center=True)
    add_para(doc, "Master of Science in Artificial Intelligence", center=True)
    add_para(doc, "", center=True)
    add_para(doc, "Aug 7, 2026", center=True)
    doc.add_page_break()

    # --- Abstract ------------------------------------------------------------
    add_para(doc, "Abstract", center=True, bold=True)
    add_body(
        doc,
        "This report documents a testing and debugging pass on TrustResume's Trust "
        "Harness, the agent that checks every claim in a generated resume against the "
        "candidate's own evidence. The project's offline evaluation harness scores the "
        "Trust Harness against 12 hand-labeled claim/evidence/verdict cases. Run "
        "against the harness's original prompt, it measured accuracy 0.500 and "
        "macro-F1 0.489, with every error running the same direction: a strictness "
        "instruction with no defined labels caused a uniform one-notch downgrade. "
        "Replacing that instruction with explicit label definitions and a named "
        "asymmetry between the two error types raised accuracy and macro-F1 to 0.833, "
        "reproduced identically across independent runs at temperature 0, with zero "
        "dangerous (too-lenient) errors before and after. Two remaining "
        "misclassifications and a related retrieval weak spot are analyzed as open "
        "findings. Separately, this pass added per-step cost telemetry and fixed a "
        "pricing-configuration gap that had been silently reporting every Bedrock "
        "generation's cost as unknown.",
    )
    doc.add_page_break()

    # --- 1. Introduction -------------------------------------------------------
    add_heading(doc, "1. Introduction")
    add_body(
        doc,
        "TrustResume's central claim is that every factual statement in a generated "
        "resume is checked against the candidate's own evidence before it is shown to "
        "anyone. That claim is only as good as the Trust Harness agent making the "
        "check, and the runtime Trust score shown to a user is computed directly from "
        "whatever the harness says — a harness that rubber-stamps every claim SUPPORTED "
        "would produce a perfect score and an invisible failure. ADR-0011 added the "
        "missing check: a small, hand-labeled dataset of claim/evidence pairs with a "
        "known correct verdict, scored offline and independent of any live user.",
    )
    add_body(
        doc,
        "This report is the testing and debugging exercise that dataset made possible. "
        "Section 2 covers the testing methodology; Section 3 reports an A/B comparison "
        "of the harness's original prompt versus a redefined one; Section 4 analyzes "
        "the errors that remain, plus a related retrieval weak spot; Section 5 covers "
        "reliability work done alongside the fix — per-step cost telemetry and a "
        "pricing-configuration bug that was silently making every real generation's "
        "cost unreportable.",
    )

    # --- 2. Testing Methodology -------------------------------------------------
    add_heading(doc, "2. Testing Methodology")
    add_body(
        doc,
        "The harness under test is TrustHarnessAgent: given a resume draft and its "
        "retrieved evidence, it extracts every discrete factual claim and classifies "
        "each as SUPPORTED, PARTIALLY_SUPPORTED, or UNSUPPORTED. The evaluation dataset "
        "(evals/datasets/trust_claims.jsonl) pins 12 claims against known evidence and a "
        "known correct label, weighted toward the SUPPORTED/PARTIALLY_SUPPORTED "
        "boundary because that is where a resume actually lies — quiet inflation of "
        "scope, seniority, and numbers — with three flat fabrications included as an "
        "easy floor.",
    )
    add_body(
        doc,
        "python -m trustresume.evals --suite trust drives every case through the real "
        "agent and model (AWS Bedrock, global.anthropic.claude-opus-4-6-v1, temperature "
        "0) and scores accuracy, macro-F1, per-label precision/recall/F1, and a full "
        "confusion matrix. Macro-F1 is reported alongside accuracy because a real "
        "resume's claims skew SUPPORTED, so a harness that never predicts UNSUPPORTED "
        "can look ~80% accurate while failing at its only job. A dangerous_errors count "
        "is tracked separately: cases where the harness was too lenient, calling an "
        "unsupported claim supported. That error ships a fabrication to an employer; "
        "the opposite error only costs one more rewrite iteration, so the two are not "
        "reported as one undifferentiated rate.",
    )
    add_body(
        doc,
        "Section 3's A/B test compares the harness's original prompt against a revised "
        "one on this same 12-case set, holding the model, temperature, and dataset "
        "fixed. Reproducibility was checked by running the current configuration twice "
        "independently; both returned identical accuracy, macro-F1, and per-case "
        "verdicts, as expected at temperature 0.",
    )

    # --- 3. A/B Test Results -----------------------------------------------------
    add_heading(doc, "3. A/B Test Results: Trust Harness Prompt")
    add_body(
        doc,
        "The harness's original system prompt said only “Be strict... do not give "
        "the draft the benefit of the doubt,” without ever defining what SUPPORTED, "
        "PARTIALLY_SUPPORTED, and UNSUPPORTED mean. Run against the 12-case dataset, "
        "that prompt (“before”) versus the redefined-labels prompt "
        "(“after”) scored:",
    )
    add_table(
        doc,
        ["Metric", "Before (original prompt)", "After (redefined labels)"],
        [
            ["Accuracy", "0.500", "0.833"],
            ["Macro-F1", "0.489", "0.833"],
            ["SUPPORTED recall", "0.25", "1.00"],
            ["Too-lenient (dangerous) errors", "0", "0"],
        ],
    )
    add_body(
        doc,
        "The per-case detail, not just the headline numbers, is what made this "
        "actionable: all six errors under the original prompt ran the same direction "
        "and by the same margin — off by exactly one severity step, always toward "
        "under-crediting. A claim reading “ran 23 postmortems over 18 months” "
        "against evidence stating the same thing verbatim was downgraded to "
        "PARTIALLY_SUPPORTED — not a model that cannot tell supported from "
        "unsupported, but one told to be cautious with no boundary on what that means, "
        "so it applied a uniform one-notch penalty to everything.",
    )
    add_body(
        doc,
        "The fix replaced the strictness instruction with explicit label definitions: "
        "SUPPORTED includes a restatement, an entailed generalization, or a correctly "
        "derived figure; PARTIALLY_SUPPORTED is reserved for real substance with "
        "overstated scope, seniority, or numbers; UNSUPPORTED is reserved for claims "
        "with no evidentiary basis at all. The prompt also names the asymmetry "
        "directly — an unsupported claim marked SUPPORTED ships a fabrication, while "
        "a supported claim marked otherwise only costs one rewrite — so the model can "
        "resist overstatement without discounting everything by default.",
    )
    add_body(
        doc,
        "Both conditions were reproduced independently in this pass rather than only "
        "quoted from an earlier commit: the after-prompt returned 0.833/0.833 on two "
        "separate runs, and temporarily restoring the original prompt returned "
        "0.500/0.489, exactly matching. Too-lenient errors held at zero in both "
        "conditions — the harness got more accurate without becoming more permissive.",
    )

    # --- 4. Error Analysis --------------------------------------------------------
    add_heading(doc, "4. Error Analysis")
    add_body(
        doc,
        "Two of the twelve cases are still misclassified after the fix, both "
        "PARTIALLY_SUPPORTED claims judged UNSUPPORTED — the safe direction, since "
        "neither lets a fabrication through, but still worth naming precisely rather "
        "than folding into an aggregate accuracy number.",
    )
    add_table(
        doc,
        ["Case", "Claim", "Evidence", "Expected", "Predicted"],
        [
            [
                "t6",
                "Owned the CI/CD platform and cut build times by 90%.",
                "Owned the CI/CD platform for 40 engineers: a shared caching layer "
                "that cut mean build time from 11 to 4 minutes.",
                "PARTIALLY_SUPPORTED",
                "UNSUPPORTED",
            ],
            [
                "t7",
                "Expert-level Kubernetes administrator.",
                "Taught myself enough of the container orchestrator to debug a "
                "scheduling bug: pods stuck pending because a node taint had no "
                "matching toleration.",
                "PARTIALLY_SUPPORTED",
                "UNSUPPORTED",
            ],
        ],
    )
    add_body(
        doc,
        "t6 is a compound claim with one true part (ownership) and one false part "
        "(11-to-4 minutes is ~64%, not 90%); the model let the false numeric "
        "component drag the whole claim to UNSUPPORTED rather than averaging the two, "
        "even though the prompt's definitions call for PARTIALLY_SUPPORTED when a "
        "compound claim's parts differ. t7 is seniority inflation: self-taught "
        "knowledge sufficient to debug one scheduling bug is escalated to "
        "“expert-level administrator.” The prompt's own definitions use "
        "almost this exact evidence shape as a PARTIALLY_SUPPORTED example, yet the "
        "model still called it baseless. Both point at the same gap: the definitions "
        "state the boundary abstractly, but a model can still resolve a genuinely "
        "borderline case toward the stricter neighbor. A concrete next step is adding "
        "more compound-claim and seniority-gap examples to the labeled dataset itself, "
        "not just the prompt, so a future revision is scored against this exact "
        "failure mode.",
    )
    add_body(
        doc,
        "A related weak spot surfaces on the retrieval side of the same suite, not the "
        "Trust Harness. One evaluation query is deliberately unanswerable (no truly "
        "relevant document exists), but hybrid retrieval has no notion of “no "
        "good answer” — it always fills its configured top-k, so the query still "
        "returns 8 irrelevant chunks. Retrieval recall is unaffected by design (a "
        "query with nothing relevant to find is correctly scored as “missed "
        "nothing”), but the writer agent still receives 8 chunks that should not "
        "ground anything. The Trust Harness is the actual safeguard here, and the "
        "zero dangerous-error result in Section 3 is direct evidence it is working — "
        "but a relevance threshold on the retrieval side would be a cheaper first line "
        "of defense than relying on the harness alone.",
    )

    # --- 5. Reliability Improvements & Best Practices -----------------------------
    add_heading(doc, "5. Reliability Improvements & Best Practices")
    add_body(
        doc,
        "Two changes made alongside the fix address reliability rather than model "
        "quality, both found by measuring a real run rather than trusting it worked. "
        "First, per-step cost and token attribution: telemetry previously recorded "
        "total tokens per model and total duration per orchestrator node as two "
        "unlinked lists, so “which step of which rewrite iteration cost how "
        "much” was unanswerable. LangGraph already tags every LLM call's callback "
        "metadata with the node that made it; the tracker now captures that tag and "
        "attributes each call's tokens and cost to it, with no agent changes required. "
        "Every run's output directory now includes a metrics.json with a per-step, "
        "per-iteration breakdown.",
    )
    add_body(
        doc,
        "Second, a pricing-configuration gap: config/pricing.json listed OpenAI rates "
        "only, so every real generation against the project's default provider (AWS "
        "Bedrock / Claude) reported cost_usd = null — technically correct behavior "
        "for an unpriced model, but Bedrock was never actually unpriced, just missing "
        "from the file. The fix adds real Claude rates plus a version-extension guard "
        "so a future model release cannot silently inherit an earlier generation's "
        "price by accidental substring match.",
    )
    add_body(
        doc,
        "Best practices worth stating explicitly: never report a partial or default "
        "number when the true value is unknown; when a classifier's errors are "
        "correlated, look at their direction and shape before touching the model or "
        "data; treat too-lenient and too-strict classification errors as different "
        "severities, not one rate; and reproduce a before/after comparison at a "
        "pinned temperature, in both directions, before treating a fix as confirmed.",
    )

    # --- 6. Conclusion --------------------------------------------------------------
    add_heading(doc, "6. Conclusion")
    add_body(
        doc,
        "The Trust Harness's offline evaluation measured a real regression (accuracy "
        "0.500) in the harness's original prompt, root-caused it to an undefined "
        "strictness instruction rather than a model or data problem, and confirmed a "
        "fix (accuracy 0.833) that closed the gap without opening the more dangerous "
        "failure mode of letting a fabrication through. Two PARTIALLY_SUPPORTED cases "
        "still misclassify as UNSUPPORTED, in the safe direction, and are documented as "
        "open findings with a concrete next step rather than treated as solved. A "
        "related retrieval weak spot is currently caught downstream by the Trust "
        "Harness rather than filtered upstream, which works today but is a thinner "
        "margin than a retrieval-side fix would be. Alongside the model-quality work, "
        "per-step cost telemetry and a pricing fix closed a gap where every real "
        "generation's cost was silently unreportable, extending the same “never "
        "report an unearned number” discipline to the system's cost-accounting path.",
    )

    # --- AI Disclosure Statement -----------------------------------------------------
    add_para(doc, "AI Disclosure Statement", bold=True)
    add_body(
        doc,
        "In August 2026, during Week 6 of this project, an AI coding assistant (Claude) "
        "was used to run the offline evaluation harness, independently reproduce the "
        "before/after A/B comparison (including temporarily restoring the original "
        "prompt to re-derive the “before” numbers rather than only quoting an "
        "earlier commit), implement the Section 5 telemetry/pricing fix, and draft this "
        "report. The Trust Harness prompt fix itself (Section 3) was implemented in an "
        "earlier session and is reported here, not newly authored. All numeric results "
        "came from actually running the project's code against a real Bedrock-hosted "
        "model. All findings and conclusions were reviewed and confirmed by the author.",
    )

    return doc


def main() -> None:
    doc = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
