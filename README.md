# TrustResume (LangChain / LangGraph / ChromaDB port)

Evidence-based, ATS-friendly resume generation using RAG, multi-agent AI, and
trust verification — an MSAI-699 capstone project.

This is a from-scratch reimplementation of
[`trustresume`](https://github.com/pxu/trustresume) (the original pydantic-ai +
Qdrant MVP) on **LangChain + LangGraph + ChromaDB**, keeping the SQLite storage
layer and the overall architecture (five/six agents, an orchestrated quality
loop, trust verification decoupled from generation) unchanged. See
`architecture/` for the living design and ADRs, and
`docs/code-walkthrough.md` for a narrative learning guide to the whole
codebase.

## Status

Backend port complete (models → storage → retrieval → ingestion → agents →
orchestration → API), plus a Streamlit frontend, CI, Docker, structured
logging, hybrid (vector + keyword) retrieval, and content-hash ingestion
dedup added on top of the original port. See `architecture/high-level-design.md`
for the current module map and `docs/code-walkthrough.md` for how it all fits
together.

## Tech stack

LangChain (`with_structured_output` agents) · LangGraph (orchestrator + quality
loop) · ChromaDB (vector storage) · SQLite + FTS5 (structured storage +
keyword search, hybrid-fused with vector search via RRF) · `unstructured`
(.docx/.pdf parsing) · FastAPI · Streamlit (frontend) · FastEmbed (embeddings)
· AWS Bedrock / OpenAI / Google — configurable LLM provider.

## Getting started

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/). `uv.lock` is
committed — install from it for reproducible versions.

```bash
uv venv --python 3.13
source .venv/bin/activate
uv sync --locked --extra dev --extra providers --extra ui

pytest                          # offline unit + integration tests (coverage gate: 95%)
pytest -m live                  # + tests hitting a real embedding model / uvicorn subprocess
ruff check .                    # lint
mypy src                        # type-check
```

Run the API (offline, no credentials needed):

```bash
TRUSTRESUME_LLM_PROVIDER=test uvicorn trustresume.api.server:build_served_app --factory --port 8000
```

Run the Streamlit frontend against it:

```bash
TRUSTRESUME_API_URL=http://localhost:8000 streamlit run src/trustresume/ui/streamlit_app.py
```

Or run both together via Docker Compose (also credential-free by default):

```bash
docker compose up --build
```

See `CLAUDE.md` for the full command reference (real LLM providers, single-test
invocation, format/lint/type-check scope) and `docs/code-walkthrough.md` for a
guided tour of the codebase.
