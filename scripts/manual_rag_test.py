"""Manual end-to-end RAG smoke test: real Bedrock, real fastembed, real Chroma/SQLite.

Ingests two sample resumes (.docx and .pdf) as one candidate's evidence, then
generates a resume tailored to a target job posting. Throwaway script — not
part of the test suite, not imported by anything. Uses isolated temp
DB/Chroma paths so it never touches the repo's default
trustresume.db/chroma_data.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from trustresume.api.app_service import build_default_app
from trustresume.api.model_factory import LLMConfig
from trustresume.ingestion.parser import parse_document
from trustresume.models import DocumentType

REPO_ROOT = Path(__file__).resolve().parents[1]
USER_ID = "manual-test-user"
DEFAULT_JOB_POSTING = REPO_ROOT / "data/sample_job_descriptions/Sample_Job_Description.docx"


def main() -> None:
    job_posting_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JOB_POSTING
    tmp_dir = tempfile.mkdtemp(prefix="trustresume_manual_")
    try:
        db_path = str(Path(tmp_dir) / "manual.db")
        chroma_path = str(Path(tmp_dir) / "manual_chroma")
        print(f"[setup] isolated db={db_path} chroma={chroma_path}", file=sys.stderr)

        with build_default_app(
            db_path=db_path, chroma_path=chroma_path, llm_config=LLMConfig.from_env()
        ) as app:
            app.ensure_user("Manual Test User", user_id=USER_ID)

            resume_files = [
                REPO_ROOT / "data/sample_documents/AI_Engineer_Resume.docx",
                REPO_ROOT / "data/sample_documents/Senior_SDE_Resume.pdf",
            ]
            for path in resume_files:
                doc_id = app._ingestion.ingest_file(
                    user_id=USER_ID, path=str(path), document_type=DocumentType.RESUME
                )
                print(f"[ingest] {path.name} -> document_id={doc_id}", file=sys.stderr)

            job_posting_text = parse_document(job_posting_path)
            print(
                f"[job] loaded {job_posting_path.name} ({len(job_posting_text)} chars)",
                file=sys.stderr,
            )

            print("[generate] running full pipeline against Bedrock...", file=sys.stderr)
            state = app.generate(user_id=USER_ID, job_posting=job_posting_text)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    result = {
        "iteration": state.iteration,
        "passed": state.passed,
        "exhausted": state.is_exhausted,
        "trust_score": state.current_trust.score if state.current_trust else None,
        "ats_score": state.current_ats.score if state.current_ats else None,
        "hallucinations": [
            {"text": c.text, "category": c.category.value}
            for c in (state.current_trust.hallucinations if state.current_trust else [])
        ],
        "missing_keywords": state.current_ats.missing_keywords if state.current_ats else [],
        "matched_keywords": state.current_ats.matched_keywords if state.current_ats else [],
        "draft": state.current_draft.model_dump() if state.current_draft else None,
        "job_title": state.job.title if state.job else None,
        "candidate_profile": state.candidate_profile.model_dump()
        if state.candidate_profile
        else None,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
