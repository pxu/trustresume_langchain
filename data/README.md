# data/

**Never commit real candidate documents here** — the design explicitly scopes
candidate data as local-only and user-isolated (see the proposal's Ethical
Considerations and ADR-0001). This folder is for synthetic/sample data only.

- `sample_documents/` — synthetic example resumes, project reports, STAR
  stories, certifications (for demos/tests, not anyone's real materials).
  **Exception**: `AI_Engineer_Resume.docx`/`Senior_SDE_Resume.pdf` are real
  personal résumés kept locally for manual RAG testing
  (`scripts/manual_rag_test.py`) — both are explicitly `.gitignore`d (see the
  "Candidate data & secrets" section there) precisely because they're real,
  not synthetic; don't remove that gitignore entry.
- `sample_job_descriptions/` — a handful of example JDs (public job postings,
  not anyone's private data) matching the personas in the design doc's
  "Adaptive Resume Personalization" section.
- `processed/` — gitignored. Local output of the ingestion pipeline.
- `embeddings/` — gitignored. Local embedding cache (vectors themselves live
  in Pinecone, a managed remote index — this is not a vector store directory).
