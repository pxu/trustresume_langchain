"""Generate the architecture diagrams embedded in docs/detailed-design.pdf.

Standalone, throwaway doc-tooling (like scripts/manual_rag_test.py) — not
part of the app, not covered by the test suite. Run:

    .venv/bin/python scripts/generate_design_diagrams.py

Regenerates every PNG under docs/diagrams/ from scratch (matplotlib only —
no pandoc/graphviz/mermaid available in this environment).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Palette — muted, colorblind-safe, print-friendly.
NAVY = "#1f3552"
BLUE = "#3f6fb0"
BLUE_LIGHT = "#dbe6f3"
TEAL = "#2f8f83"
TEAL_LIGHT = "#dcf0ec"
AMBER = "#c9871f"
AMBER_LIGHT = "#f7e6cc"
GRAY = "#5b6472"
GRAY_LIGHT = "#eceff2"
RED = "#b0463f"
WHITE = "#ffffff"

plt.rcParams["font.family"] = "DejaVu Sans"


def box(
    ax,
    xy,
    w,
    h,
    text,
    *,
    face=BLUE_LIGHT,
    edge=BLUE,
    fontsize=10,
    fontweight="normal",
    textcolor=NAVY,
    linewidth=1.4,
    linestyle="solid",
    zorder=2,
):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=textcolor,
        zorder=zorder + 1,
        wrap=True,
    )
    return patch


def arrow(ax, start, end, *, color=GRAY, style="-|>", lw=1.4, connectionstyle="arc3,rad=0.0", ls="solid"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
        linestyle=ls,
        zorder=1,
    )
    ax.add_patch(patch)
    return patch


def new_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# 1. System overview / component diagram
# ---------------------------------------------------------------------------
def diagram_system_overview():
    fig, ax = new_fig(13, 9.2)

    box(ax, (0.6, 7.9), 3.2, 0.9, "Streamlit UI\n(ui/)\nthin REST client", face=GRAY_LIGHT, edge=GRAY)
    arrow(ax, (2.2, 7.9), (2.2, 7.1), color=GRAY)
    ax.text(2.75, 7.5, "HTTP", fontsize=8.5, color=GRAY, style="italic")

    box(ax, (0.6, 6.2), 3.2, 0.9, "FastAPI\n(api/server.py)\nroutes only, no logic", face=BLUE_LIGHT, edge=BLUE)
    arrow(ax, (2.2, 6.2), (2.2, 5.4), color=BLUE)

    box(
        ax,
        (0.6, 4.5),
        3.2,
        0.9,
        "TrustResumeApp\n(api/app_service.py)\napplication facade",
        face=BLUE_LIGHT,
        edge=BLUE,
        fontweight="bold",
    )
    arrow(ax, (2.2, 4.5), (2.2, 3.7), color=BLUE)

    box(
        ax,
        (0.6, 2.8),
        3.2,
        0.9,
        "Orchestrator\n(orchestration/)\nLangGraph StateGraph",
        face=TEAL_LIGHT,
        edge=TEAL,
        fontweight="bold",
    )

    # One arrow from the orchestrator to a bracket spanning the agent column —
    # six individual curved arrows were visually noisy and crossed the boxes.
    agent_x = 5.4
    agent_top = 8.7
    agent_w, agent_h, gap = 3.4, 0.82, 0.22
    agent_labels = [
        "JobDescriptionAgent  (LLM)",
        "CandidateProfileAgent  (LLM, cached)",
        "EvidenceRetrievalAgent  (no LLM)",
        "ResumeWriterAgent  (LLM)",
        "TrustHarnessAgent  (LLM)",
        "ATSEvaluationAgent  (no LLM)",
    ]
    agent_bottoms = []
    for i, label in enumerate(agent_labels):
        y_top = agent_top - i * (agent_h + gap)
        face = TEAL_LIGHT if "no LLM" in label else AMBER_LIGHT
        edge = TEAL if "no LLM" in label else AMBER
        box(ax, (agent_x, y_top - agent_h), agent_w, agent_h, label, face=face, edge=edge, fontsize=9.3)
        agent_bottoms.append(y_top - agent_h / 2)

    bracket_x = agent_x - 0.35
    arrow(ax, (3.8, 3.25), (bracket_x, 3.25), color=TEAL, lw=1.3)
    ax.annotate(
        "",
        xy=(bracket_x, agent_bottoms[0] + 0.3),
        xytext=(bracket_x, agent_bottoms[-1] - 0.3),
        arrowprops={"arrowstyle": "-", "color": TEAL, "lw": 1.3},
    )
    ax.plot([bracket_x, agent_x], [3.25, 3.25], color=TEAL, lw=1.3)

    ax.text(
        agent_x + agent_w / 2,
        9.05,
        "agents/  — six pure input → output steps, sequenced by the orchestrator",
        ha="center",
        fontsize=9.8,
        color=NAVY,
        style="italic",
    )

    # Storage layer at the bottom
    arrow(ax, (2.2, 2.8), (2.2, 2.05), color=TEAL)
    box(
        ax,
        (0.6, 1.0),
        3.2,
        0.9,
        "SQLite\n(storage/)\nusers, docs, jobs, chunks,\nresumes, evaluations",
        face=AMBER_LIGHT,
        edge=AMBER,
        fontsize=9,
    )
    box(
        ax,
        (4.3, 1.0),
        3.6,
        0.9,
        "Chroma + FTS5\n(retrieval/)\nvector + keyword,\nfused by RRF (hybrid)",
        face=AMBER_LIGHT,
        edge=AMBER,
        fontsize=9,
    )
    arrow(ax, (3.8, 1.45), (4.3, 1.45), color=AMBER, style="<|-|>")
    ax.text(4.05, 1.75, "user_id +\nchunk_id", fontsize=7.5, color=GRAY, ha="center")

    box(
        ax,
        (9.1, 4.1),
        3.5,
        1.0,
        "ingestion/\nparse → clean → dedup → chunk\n→ write both stores",
        face=GRAY_LIGHT,
        edge=GRAY,
        fontsize=9,
    )
    ax.text(10.85, 5.35, "independent write path\n(document upload/delete)", fontsize=8, ha="center", color=GRAY)
    arrow(ax, (9.6, 4.1), (7.9, 1.9), color=GRAY, lw=1.1, connectionstyle="arc3,rad=-0.25")
    arrow(ax, (9.4, 4.05), (4.5, 1.45), color=GRAY, lw=1.1, connectionstyle="arc3,rad=0.3")

    box(
        ax,
        (9.1, 2.6),
        3.5,
        1.0,
        "trust_verification/\nevaluation/\npure scoring functions,\nno LLM/framework import",
        face=GRAY_LIGHT,
        edge=GRAY,
        fontsize=9,
    )

    ax.set_title(
        "TrustResume — System Overview\nUI → HTTP → Facade → Orchestrator → Agents → Storage",
        fontsize=13.5,
        color=NAVY,
        fontweight="bold",
        pad=14,
    )
    save(fig, "01_system_overview.png")


# ---------------------------------------------------------------------------
# 2. LangGraph orchestrator state diagram
# ---------------------------------------------------------------------------
def diagram_orchestrator_graph():
    fig, ax = new_fig(12, 5.4)

    nodes = [
        ("START", 0.6, GRAY_LIGHT, GRAY),
        ("analyze_job", 2.1, BLUE_LIGHT, BLUE),
        ("load_candidate_\nprofile", 3.9, BLUE_LIGHT, BLUE),
        ("retrieve_\nevidence", 5.7, BLUE_LIGHT, BLUE),
        ("write_resume", 7.5, AMBER_LIGHT, AMBER),
        ("score_trust", 9.3, AMBER_LIGHT, AMBER),
        ("score_ats", 11.1, AMBER_LIGHT, AMBER),
    ]
    y = 3.6
    w, h = 1.55, 0.85
    centers = []
    for label, x, face, edge in nodes:
        box(ax, (x - w / 2, y - h / 2), w, h, label, face=face, edge=edge, fontsize=8.7)
        centers.append((x, y))
    for (x1, y1), (x2, y2) in zip(centers, centers[1:], strict=False):
        arrow(ax, (x1 + w / 2, y1), (x2 - w / 2, y2), color=GRAY)

    # once-only bracket over analyze_job..retrieve_evidence
    ax.annotate(
        "",
        xy=(2.1 - w / 2, y + h / 2 + 0.55),
        xytext=(5.7 + w / 2, y + h / 2 + 0.55),
        arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 1.2},
    )
    ax.text(3.9, y + h / 2 + 0.75, "run once per generation", fontsize=9, ha="center", color=GRAY, style="italic")

    # quality-loop bracket over write_resume..score_ats
    ax.annotate(
        "",
        xy=(7.5 - w / 2, y + h / 2 + 0.55),
        xytext=(11.1 + w / 2, y + h / 2 + 0.55),
        arrowprops={"arrowstyle": "-", "color": AMBER, "lw": 1.2},
    )
    ax.text(9.3, y + h / 2 + 0.75, "quality loop (repeatable)", fontsize=9, ha="center", color=AMBER, style="italic")

    # conditional edge from score_ats
    end_x, end_y = 11.6, 1.6
    box(ax, (end_x - 0.7, end_y - 0.4), 1.4, 0.8, "END", face=GRAY_LIGHT, edge=GRAY, fontsize=9)
    arrow(
        ax,
        (11.1, y - h / 2),
        (end_x, end_y + 0.4),
        color="#2f8f4f",
        connectionstyle="arc3,rad=-0.15",
    )
    ax.text(11.55, 2.85, "iter ≥ 3", fontsize=8, color="#2f8f4f", ha="left")
    ax.text(11.35, 2.4, "(ships best draft\nby final_index, not\njust the last one)", fontsize=7, color="#2f8f4f", ha="left", style="italic")

    rewrite_x, rewrite_y = 8.4, 1.6
    box(
        ax,
        (rewrite_x - 1.0, rewrite_y - 0.4),
        2.0,
        0.8,
        "prepare_rewrite\n(iteration += 1)",
        face="#fbe3e0",
        edge=RED,
        fontsize=8.7,
    )
    arrow(ax, (9.3, y - h / 2), (rewrite_x + 0.3, rewrite_y + 0.4), color=RED, connectionstyle="arc3,rad=0.15")
    ax.text(9.9, 2.55, "else: rewrite\n(always — even\nif already passed)", fontsize=8, color=RED, ha="left")
    arrow(
        ax,
        (rewrite_x - 1.0, rewrite_y),
        (7.5 - w / 2, y - h / 2 - 0.05),
        color=RED,
        connectionstyle="arc3,rad=0.35",
    )
    ax.text(6.0, 1.1, "back to write_resume", fontsize=8, color=RED, ha="center")

    ax.text(
        6.0,
        0.15,
        "_route() reads `iteration` BEFORE prepare_rewrite increments it, and (ADR-0016) no longer checks "
        "`passed` — default max_iterations=3 ⇒ iterations 0,1,2,3 all run, always, ⇒ exactly 4 total drafts.",
        fontsize=9,
        ha="center",
        color=NAVY,
        style="italic",
        wrap=True,
    )

    ax.set_title(
        "Orchestrator — LangGraph StateGraph\n(orchestration/orchestrator.py)",
        fontsize=13,
        color=NAVY,
        fontweight="bold",
        pad=10,
    )
    save(fig, "02_orchestrator_graph.png")


# ---------------------------------------------------------------------------
# 3. Generation sequence (end-to-end data flow)
# ---------------------------------------------------------------------------
def diagram_generation_sequence():
    fig, ax = new_fig(11.5, 10.6)

    steps = [
        ("1", "JobDescriptionAgent.run(posting)", "→ JobDescription", "once", BLUE_LIGHT, BLUE),
        ("2", "CandidateProfileService.get_or_refresh(uid)", "→ CandidateProfile (cache hit usually)", "once", BLUE_LIGHT, BLUE),
        ("3", "EvidenceRetrievalAgent.run(uid, job)", "→ EvidenceSet (hybrid, user-scoped)", "once", BLUE_LIGHT, BLUE),
        ("4", "ResumeWriterAgent.run(job, evidence, feedback?)", "→ ResumeDraft", "loop", AMBER_LIGHT, AMBER),
        ("5", "TrustHarnessAgent.run(draft, evidence)", "→ TrustReport (LLM classifies, code scores)", "loop", AMBER_LIGHT, AMBER),
        ("6", "ATSEvaluationAgent.run(draft, job)", "→ ATSReport (deterministic coverage)", "loop", AMBER_LIGHT, AMBER),
        ("7", "TrustResumeApp._persist(state)", "→ SQLite: best draft (final_index) + PDF/MD export + scores", "once", TEAL_LIGHT, TEAL),
        ("8", "WorkflowState → GenerateResponse", "real scores + flagged claims + missing keywords", "once", TEAL_LIGHT, TEAL),
    ]

    top = 10.0
    row_h = 1.05
    for i, (num, title, detail, tag, face, edge) in enumerate(steps):
        y = top - i * row_h
        box(ax, (0.3, y - 0.35), 0.7, 0.7, num, face=edge, edge=edge, fontsize=13, fontweight="bold", textcolor="white")
        box(ax, (1.3, y - 0.38), 7.7, 0.76, "", face=face, edge=edge, linewidth=1.0)
        ax.text(1.55, y + 0.12, title, fontsize=9.8, color=NAVY, fontweight="bold", va="center")
        ax.text(1.55, y - 0.16, detail, fontsize=8.6, color=GRAY, va="center")
        box(ax, (9.2, y - 0.22), 1.0, 0.44, tag, face="white", edge=edge, fontsize=8, textcolor=edge)
        if i < len(steps) - 1:
            arrow(ax, (0.65, y - 0.35), (0.65, y - row_h + 0.35), color=GRAY, lw=1.2)

    # quality-loop bracket around steps 4-6, well clear of the "loop" tags
    loop_top = top - 3 * row_h + 0.5
    loop_bot = top - 5 * row_h - 0.5
    bracket_x = 10.55
    ax.annotate(
        "",
        xy=(bracket_x, loop_top),
        xytext=(bracket_x, loop_bot),
        arrowprops={"arrowstyle": "-", "color": AMBER, "lw": 1.4},
    )
    ax.text(
        bracket_x + 0.35,
        (loop_top + loop_bot) / 2,
        "quality\nloop\nexactly 4\npasses,\nalways",
        fontsize=8.5,
        color=AMBER,
        ha="left",
        va="center",
    )

    ax.text(
        4.6,
        top - 7 * row_h - 1.05,
        "At iteration 3 already  →  stop (ADR-0016: passing does NOT stop the loop early).\n"
        "Otherwise  →  build_feedback(trust, ats)  →  back to step 4, regardless of pass/fail.\n"
        "Once stopped: ship the best-scoring draft (passed > failed, then higher ATS) — not just the last one.",
        fontsize=9,
        ha="center",
        va="top",
        color=NAVY,
        style="italic",
    )

    ax.set_title(
        "One POST /api/generate — End-to-End Data Flow",
        fontsize=13.5,
        color=NAVY,
        fontweight="bold",
        pad=14,
    )
    save(fig, "03_generation_sequence.png")


# ---------------------------------------------------------------------------
# 4. Storage schema (ER-style)
# ---------------------------------------------------------------------------
def diagram_storage_schema():
    fig, ax = new_fig(12, 8.6)

    def table(ax, xy, w, h, title, cols, *, edge=BLUE, face=BLUE_LIGHT):
        x, y = xy
        box(ax, (x, y), w, h, "", face=face, edge=edge, linewidth=1.3)
        ax.text(x + w / 2, y + h - 0.28, title, fontsize=9.6, fontweight="bold", ha="center", color=NAVY)
        ax.plot([x + 0.05, x + w - 0.05], [y + h - 0.42, y + h - 0.42], color=edge, lw=0.9)
        for i, col in enumerate(cols):
            ax.text(x + 0.15, y + h - 0.62 - i * 0.24, col, fontsize=7.6, color=GRAY, va="center")

    users = ("users", ["id (PK)", "name", "created_at"])
    documents = (
        "documents",
        ["id (PK)", "user_id (FK)", "filename", "document_type", "content_hash", "created_at"],
    )
    jobs = (
        "jobs",
        ["id (PK)", "user_id (FK)", "title / company / summary", "raw_posting", "job_description_json"],
    )
    job_documents = ("job_documents", ["job_id (FK)", "document_id (FK)", "created_at"])
    chunks = (
        "chunks",
        ["chunk_id (PK)", "user_id (FK)", "document_id (FK)", "chunk_index", "text"],
    )
    chunks_fts = ("chunks_fts (FTS5)", ["text", "external content=chunks", "kept in sync by triggers"])
    resumes = (
        "generated_resumes",
        ["id (PK)", "user_id (FK)", "job_id (FK, SET NULL)", "trust_score / ats_score", "pdf_bytes / markdown_text"],
    )
    evaluations = (
        "evaluations",
        ["id (PK)", "resume_id (FK)", "user_id (FK)", "trust_report_json", "ats_report_json"],
    )
    profiles = ("candidate_profiles", ["user_id (PK)", "profile_json", "doc_hash", "stale"])

    table(ax, (0.4, 7.1), 2.6, 1.3, *users, edge=NAVY, face=GRAY_LIGHT)
    table(ax, (0.4, 4.7), 3.0, 1.9, *documents)
    table(ax, (4.0, 4.7), 3.0, 1.9, *jobs, edge=TEAL, face=TEAL_LIGHT)
    table(ax, (4.0, 3.1), 3.0, 1.3, *job_documents, edge=TEAL, face=TEAL_LIGHT)
    table(ax, (0.4, 2.2), 3.0, 1.9, *chunks)
    table(ax, (0.4, 0.3), 3.0, 1.5, *chunks_fts, edge=AMBER, face=AMBER_LIGHT)
    table(ax, (7.6, 4.7), 3.6, 2.2, *resumes, edge=RED, face="#fbe3e0")
    table(ax, (7.6, 2.2), 3.6, 1.9, *evaluations, edge=RED, face="#fbe3e0")
    table(ax, (7.6, 0.3), 3.6, 1.3, *profiles, edge=GRAY, face=GRAY_LIGHT)

    # relationships
    arrow(ax, (1.7, 7.1), (1.7, 6.6), color=GRAY, style="-", connectionstyle="arc3,rad=0")
    arrow(ax, (2.9, 7.1), (5.3, 6.6), color=GRAY, style="-", connectionstyle="arc3,rad=-0.15")
    arrow(ax, (2.9, 7.15), (9.2, 6.9), color=GRAY, style="-", connectionstyle="arc3,rad=-0.1")
    arrow(ax, (1.9, 4.7), (1.9, 4.1), color=GRAY, style="-")
    arrow(ax, (5.5, 4.7), (5.5, 4.4), color=TEAL, style="-")
    arrow(ax, (3.4, 3.8), (4.0, 3.8), color=TEAL, style="-")
    arrow(ax, (1.9, 2.2), (1.9, 1.8), color=AMBER, style="-")
    arrow(ax, (5.5, 3.1), (9.4, 5.5), color=RED, style="-", connectionstyle="arc3,rad=0.2")
    ax.text(6.3, 4.55, "SET NULL\non job delete", fontsize=7.3, color=RED, ha="center", style="italic")
    arrow(ax, (9.4, 4.7), (9.4, 4.1), color=RED, style="-")

    ax.text(
        6.0,
        8.35,
        "storage/schema.py — every table (except users) carries user_id; isolation enforced by always filtering on it (ADR-0001).",
        fontsize=9,
        ha="center",
        color=NAVY,
        style="italic",
    )
    ax.set_title("SQLite Schema", fontsize=13.5, color=NAVY, fontweight="bold", pad=6)
    save(fig, "04_storage_schema.png")


# ---------------------------------------------------------------------------
# 5. Deployment topology
# ---------------------------------------------------------------------------
def diagram_deployment():
    fig, ax = new_fig(11, 7.4)

    # Docker Compose outer boundary
    box(ax, (0.4, 0.4), 10.2, 6.2, "", face="none", edge=GRAY, linestyle="dashed", linewidth=1.6)
    ax.text(0.7, 6.35, "docker-compose.yml", fontsize=10, color=GRAY, fontweight="bold")

    # api service
    box(ax, (0.9, 3.2), 4.2, 2.6, "", face=BLUE_LIGHT, edge=BLUE, linewidth=1.6)
    ax.text(3.0, 5.55, "api  (port 8000)", fontsize=10.5, fontweight="bold", color=NAVY, ha="center")
    ax.text(
        3.0,
        4.4,
        "uvicorn\nserver:build_served_app\n\nDockerfile target: runtime\nTRUSTRESUME_LLM_PROVIDER=test\n(default; override for real creds)",
        fontsize=8.6,
        ha="center",
        va="center",
        color=NAVY,
    )

    # ui service
    box(ax, (6.1, 3.2), 4.0, 2.6, "", face=TEAL_LIGHT, edge=TEAL, linewidth=1.6)
    ax.text(8.1, 5.55, "ui  (port 8501)", fontsize=10.5, fontweight="bold", color=NAVY, ha="center")
    ax.text(
        8.1,
        4.4,
        "streamlit run\nstreamlit_app.py\n\nDockerfile target: ui\n(FROM runtime AS ui,\nshares the same venv)",
        fontsize=8.6,
        ha="center",
        va="center",
        color=NAVY,
    )

    arrow(ax, (6.1, 4.5), (5.1, 4.5), color=TEAL, style="<|-")
    ax.text(5.6, 4.75, "HTTP\nTRUSTRESUME_API_URL", fontsize=7.6, ha="center", color=TEAL)
    ax.text(5.6, 3.85, "depends_on:\napi (service_healthy)", fontsize=7.3, ha="center", color=GRAY, style="italic")

    # volume
    box(ax, (0.9, 1.0), 4.2, 1.5, "trustresume-data\n(named volume)\n/data → trustresume.db + chroma_data", face=AMBER_LIGHT, edge=AMBER, fontsize=8.8)
    arrow(ax, (3.0, 3.2), (3.0, 2.5), color=AMBER)

    # healthcheck
    ax.text(
        8.1,
        1.6,
        "healthcheck (api):\nGET /api/health every 10s\n→ gates ui's startup",
        fontsize=8.4,
        ha="center",
        color=GRAY,
        style="italic",
    )

    ax.set_title(
        "Deployment — Docker Compose\n(same multi-stage image; `runtime` and `ui` differ only in CMD)",
        fontsize=13,
        color=NAVY,
        fontweight="bold",
        pad=12,
    )
    save(fig, "05_deployment.png")


if __name__ == "__main__":
    diagram_system_overview()
    diagram_orchestrator_graph()
    diagram_generation_sequence()
    diagram_storage_schema()
    diagram_deployment()
    print("done")
