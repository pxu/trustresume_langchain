# TrustResume (LangChain / LangGraph / ChromaDB port)

Evidence-based, ATS-friendly resume generation using RAG, multi-agent AI, and
trust verification — an MSAI-699 capstone project.

This is a from-scratch reimplementation of
[`trustresume`](https://github.com/pxu/trustresume) (the original pydantic-ai +
Qdrant MVP) on **LangChain + LangGraph + ChromaDB**, keeping the SQLite storage
layer and the overall architecture (five/six agents, an orchestrated quality
loop, trust verification decoupled from generation) unchanged. See
`docs/architecture/` for the living design and ADRs, and
`docs/code-walkthrough.md` for a narrative learning guide to the whole
codebase.

## Status

Backend port complete (models → storage → retrieval → ingestion → agents →
orchestration → API), plus a Streamlit frontend, CI, Docker, structured
logging, hybrid (vector + keyword) retrieval, content-hash ingestion dedup,
persisted jobs with job-scoped retrieval, and résumé export added on top of
the original port — and, on top of that, a measurement layer: an offline
evaluation harness with labeled ground truth, per-run token/cost/latency
accounting, role-tiered models, and per-request user identity
(ADRs 0011–0014); opt-in durable execution via LangGraph checkpointing
(ADR-0015); and a redesigned quality loop that no longer stops the instant a
draft passes — it always runs a config/env-driven number of rewrites
(`config/quality_gate.json`, default 1) and ships whichever draft actually
scored best (ADR-0016). See `docs/architecture/high-level-design.md`
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

Measure retrieval quality (no credentials needed — fastembed runs locally):

```bash
python -m trustresume.evals --suite retrieval
```

Current baseline: recall@8 **1.000**, MRR **0.938** (k=8 = the depth a real generation retrieves) over a labeled 10-document
corpus — including one query whose answer shares no vocabulary with it (vector
search's job) and one naming a product the embedder treats as interchangeable
with its competitors (keyword search's job). That's the empirical case for
hybrid retrieval. See `evals/README.md`.

Run the API (offline, no credentials needed):

```bash
TRUSTRESUME_LLM_PROVIDER=test uvicorn trustresume.api.server:build_served_app --factory --port 8000
```

Or against a real model — inherits AWS credentials from your shell (default
provider is Bedrock, profile `twdc-bedrock-central`; see
`api/model_factory.py`'s `LLMConfig` for OpenAI/Google instead):

```bash
source .venv/bin/activate
uvicorn trustresume.api.server:build_served_app --factory --port 8000
```

Run the Streamlit frontend against either one, then open
[http://localhost:8501](http://localhost:8501):

```bash
TRUSTRESUME_API_URL=http://localhost:8000 streamlit run src/trustresume/ui/streamlit_app.py
```

Or run both together via Docker Compose (credential-free by default — it
always uses the offline `test` provider unless you override
`TRUSTRESUME_LLM_PROVIDER`/`TRUSTRESUME_AWS_PROFILE` *and* mount your AWS
credentials into the container, so for a quick real-model check the two
commands above are simpler):

```bash
docker compose up --build
```

See `CLAUDE.md` for the full command reference (real LLM providers, single-test
invocation, format/lint/type-check scope) and `docs/code-walkthrough.md` for a
guided tour of the codebase.
