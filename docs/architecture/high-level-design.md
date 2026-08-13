# TrustResume (LangChain/LangGraph/Chroma port) — High-Level Design

The system as ported: major components, how data flows through a generation,
and the decisions that shape the whole. This is a from-scratch reimplementation
of [`trustresume`](https://github.com/pxu/trustresume) (pydantic-ai + Qdrant)
on LangChain + LangGraph + ChromaDB, keeping the same overall architecture and
the SQLite storage layer unchanged. Read the original's `docs/architecture/` docs
for the fuller design rationale (requirements, ADRs 0002/0004–0009 all still
apply here); this doc covers what's the same, what changed, and why —
including additions made after the initial port that don't exist in the
original at all (hybrid retrieval, ingestion dedup, persisted jobs, résumé
export, a Streamlit frontend, CI/Docker, and the measurement layer of
ADRs 0011-0014: an offline eval harness, token/cost/latency telemetry,
role-tiered models, and per-request user identity).

## System overview

```
              ┌─────────────────────────────┐        ┌───────────────────────┐
              │  Streamlit UI (ui/)          │──HTTP─▶│  FastAPI backend (api/) │
              │  thin REST client only       │        │  ─ server.py (routes)   │
              └─────────────────────────────┘        │  ─ TrustResumeApp (facade)│
                                                       └─────────────┬───────────┘
                                                                     │ drives
                                                       ┌─────────────▼───────────────┐
                                                       │  Orchestrator (LangGraph)    │
                                                       │  StateGraph owns the graph   │
                                                       │  state + the quality loop     │
                                                       └──┬───┬───┬───┬───┬───────────┘
                                   sequences ─────────────┘   │   │   │   └───────────────┐
                                             ┌────────▼┐ ┌▼──────┐ ┌▼───────┐ ┌▼─────────┐
                                             │  Job   │ │Retriev│ │ Resume │ │  Trust   │ │ ATS  │
                                             │ Desc   │ │  al   │ │ Writer │ │ Harness  │ │ Eval │
                                             └───┬────┘ └──┬────┘ └───┬────┘ └────┬─────┘ └──┬───┘
                                                 │         │          │           │          │
                                                 │    ┌────▼──────────▼───────────▼──────────▼┐
                                                 │    │  shared models (unchanged) pass between all │
                                                 │    └─────────────────────────────────────┬─┘
                                       ┌─────────▼───────────┐              ┌───────────────▼────────┐
                                       │  ingestion (M3)     │              │ trust_verification/    │
                                       │  write path         │              │ evaluation (unchanged) │
                                       │  + content-hash dedup│              └────────────────────────┘
                                       └──────┬───────┬──────┘
                                       ┌──────▼──┐ ┌──▼─────────────────┐
                                       │ SQLite  │ │  Chroma + FTS5      │  hybrid store (ADR-0001)
                                       │ storage │ │  vector + keyword,  │  both keyed by user_id,
                                       │         │ │  fused by RRF        │  hybrid retrieval (ADR-0010)
                                       └─────────┘ └─────────────────────┘
```

A sixth agent, `CandidateProfileAgent`, isn't pictured — it's job-independent
and cached, invoked through `CandidateProfileService` rather than directly by
the orchestrator, exactly as in the original. The Evidence Retrieval agent
queries a `HybridRetriever` (vector + keyword, ADR-0010), not Chroma directly.

## Components

| Layer | Package | What changed from the original |
|---|---|---|
| **Frontend** | `ui/` | New — not in the original at all. Streamlit app (`streamlit_app.py`) + a thin REST client (`api_client.py`'s `TrustResumeClient`); imports nothing from `trustresume.api`/`orchestration`, so the dependency points one way (UI → HTTP → backend) and `streamlit` stays an optional (`ui`) extra. |
| **HTTP** | `api/server.py` | Routes unchanged in spirit, extended into four resource groups: documents (JSON + multipart upload + delete), jobs (CRUD, job-scoped upload/generate/list), generation and résumés (including PDF/Markdown export, and an opt-in `POST /api/runs/{run_id}/resume` for durable execution — ADR-0015), and retrieval/ops (`/api/search`, `/api/ping`, `/api/health`). Every user-scoped route resolves its caller from `X-User-Id` via one dependency (ADR-0014) instead of a hardcoded demo id. Still operates on `WorkflowState`/the facade, not agent internals. |
| **Facade** | `api/app_service.py` | Collaborator types: `HybridRetriever` (not `ChromaVectorStore` directly) feeds `EvidenceRetrievalAgent`; a LangChain `BaseChatModel` instead of a pydantic-ai `Model`. Also exposes `add_document_bytes`/`search_evidence` for the API layer. |
| **LLM provider factory** | `api/model_factory.py` | Same `LLMConfig` + precedence logic (env > llm.local.json > llm.json > default), extended with a pinned `temperature` (default 0) and per-role model tiering — `extraction`/`writer`/`verifier` (ADR-0013); provider dispatch retargeted to `langchain_aws`/`langchain_openai`/`langchain_google_genai`. The offline `"test"` provider is a new, more capable `AutoStructuredFakeChatModel` (`api/test_provider.py`) that synthesizes valid structured output for *any* bound schema, so `TRUSTRESUME_LLM_PROVIDER=test` can drive the full pipeline with no credentials (the original's pydantic-ai `TestModel` had this "synthesize anything" ability natively; LangChain's fakes don't, so it's rebuilt here). |
| **Control** | `orchestration/orchestrator.py` | Rebuilt as a LangGraph `StateGraph` (ADR-0003, superseding the original's ADR-0003). The port itself changed no public contract; `run()` has since grown `job`/`job_id`/`document_ids` for the persisted-job path (exactly one of `job_posting` or `job` must be passed) and now returns `WorkflowState.usage` — per-node timings plus tokens/cost (ADR-0012). An opt-in `checkpoint_path` (ADR-0015, off by default) enables LangGraph checkpointing so a crashed run can `resume(run_id=...)` from its last completed node; disabled, the module behaves exactly as before. `feedback.py`, `candidate_profile_service.py` unchanged. Logs each node transition and the quality-gate routing decision (`logging_config.py`, new). |
| **Workers** | `agents/` | Each LLM-backed agent swaps a pydantic-ai `Agent` for `chat_model.with_structured_output(Schema)`. `ResumeWriterAgent` binds a lenient private `_DraftExtraction` schema rather than `ResumeDraft` directly — real models emit structural noise (empty section headings, the summary emitted as its own section, bare group-label sections) that would otherwise crash structured-output parsing; `_to_resume_draft` cleans it up in code. `TrustHarnessAgent`'s private `_ClaimExtraction` schema is unchanged from the original port. `EvidenceRetrievalAgent` now takes any retriever satisfying a `search(user_id, query, limit) -> EvidenceSet` protocol (`ChromaVectorStore` or `HybridRetriever`), not `ChromaVectorStore` specifically. `ATSEvaluationAgent` unchanged. |
| **Domain logic** | `trust_verification/`, `evaluation/` | Unchanged — pure string-building and scoring functions, no framework dependency. |
| **Write path** | `ingestion/` | Parsing (`.docx`/`.pdf`) now goes through `unstructured.partition.auto.partition` (one shared entry point, replacing separate python-docx/pypdf functions); chunking uses LangChain's `RecursiveCharacterTextSplitter`. New: content-hash dedup — `IngestionService.ingest_text` checks `DocumentRepository.find_by_content_hash` before writing anything, so re-uploading the same document (by cleaned-text hash) is a no-op that returns the existing document id rather than duplicating chunks in both stores. The write-then-upsert, roll-back-on-failure contract for genuinely new documents is unchanged. |
| **Data** | `storage/` (SQLite), `retrieval/` (Chroma + FTS5) | `storage/` ported byte-for-byte, plus: `documents.content_hash` + a `UNIQUE(user_id, content_hash)` index (ingestion dedup), a `chunks_fts` FTS5 virtual table + sync triggers (keyword search), the `jobs`/`job_documents` tables (persisted jobs + job-scoped retrieval), and export/rejection/usage columns on `generated_resumes` (ADR-0012). `retrieval/` rebuilt on Chroma (ADR-0001) *and* extended with `HybridRetriever` (ADR-0010), which fuses `ChromaVectorStore.search` with `ChunkRepository.search_keywords` via Reciprocal Rank Fusion. `embedder.py` implements `langchain_core.embeddings.Embeddings`; `vector_store.py`'s `ChromaVectorStore` keeps the same three-method surface (`upsert_chunks`/`search`/`delete_chunks`) as the original's `QdrantVectorStore`. `query.py` unchanged. |
| **Contracts** | `models/` | Unchanged, plus `usage.py` (`RunUsage`/`ModelUsage`/`NodeTiming`, ADR-0012) — still pure pydantic, no framework dependency anywhere in this package. |
| **Measurement** | `evals/`, `telemetry.py` | New — nothing equivalent in the original. `evals/` scores the *system* against labeled ground truth offline (retrieval recall/MRR, Trust Harness classification accuracy — ADR-0011, `evals/README.md`); `telemetry.py` captures tokens/cost/latency per run via a LangChain callback (ADR-0012). Note the deliberate name split: `evaluation/` scores one résumé for the user at runtime, `evals/` scores the system for the engineer offline. |
| **Observability** | `logging_config.py` | New — not in the original. Stdlib `logging` + a `JsonFormatter` (one JSON object per line); `configure_logging()` runs once, at `server.py`'s `build_served_app` (the real process entry point), never from library code. |

## The generation data flow

Unchanged in shape from the original (`POST /api/generate`, with documents
already ingested):

```
1. job_description_agent.run(posting)          → JobDescription  (once)
2. candidate_profile_service.get_or_refresh(user_id) → CandidateProfile (once; usually a cache hit)
3. retrieval_agent.run(user_id, job)            → EvidenceSet     (once; hybrid vector+keyword, user-filtered)
   ┌── quality loop (≤ 4 passes: initial + 3 rewrites) ─────────────────────┐
4. resume_agent.run(job, evidence, feedback?)  → ResumeDraft
5. trust_agent.run(draft, evidence)             → TrustReport  (LLM classifies, code scores)
6. evaluation_agent.run(draft, job)             → ATSReport    (deterministic coverage)
   ── if Trust ≥ 90 AND ATS ≥ 85 → PASS, stop
   ── else if iteration == 3 → CAPPED, stop (export anyway)
   ── else build_feedback(trust, ats) → rewrite (back to step 4)
   └─────────────────────────────────────────────────────────────────────────┘
7. persist final draft + evaluation to SQLite
8. project WorkflowState → GenerateResponse (real scores + flagged claims)
```

Steps 1–6 now execute as LangGraph nodes/edges rather than a hand-rolled
`while` loop (ADR-0003), but the sequencing, the one-time-vs-per-rewrite
split, and the exact iteration-counting semantics are unchanged. Step 3's
retrieval is now hybrid (ADR-0010) rather than vector-only, but the shape
(`EvidenceRetrievalAgent.run(user_id, job) -> EvidenceSet`) the orchestrator
depends on is identical either way.

A standalone `POST /api/search` runs the same hybrid retrieval outside a full
generation, for inspecting retrieval quality directly.

With durable execution enabled (ADR-0015, opt-in via `TRUSTRESUME_CHECKPOINT_PATH`),
each run is checkpointed per node under a server-minted `run_id`, and
`POST /api/runs/{run_id}/resume` continues a crashed run from its last completed
step (steps 1–6) rather than re-running — and re-paying for — the nodes that
already succeeded. Off by default, it changes nothing about the flow above.

## Testing model

Same offline-first philosophy as the original (NFR-5): the whole stack runs
without network access or credentials.

| Real dependency | Test double |
|---|---|
| SQLite file | `connect(":memory:")` (unchanged) |
| Chroma | `chromadb.EphemeralClient()` — unique collection name per test/app instance (`TrustResumeApp`'s `chroma_collection_name` param) |
| SQLite FTS5 keyword search | Runs for real against `connect(":memory:")` — FTS5 needs no external service, so there's no fake for it |
| Embedding model | `FakeEmbeddings` (`tests/fakes.py`) — same SHA-256 hashing as the original's `FakeEmbedder`; the real `FastEmbedEmbeddings`'s lazy-load contract is unit-tested by mocking `fastembed.TextEmbedding`, its real output by a `live`-marked test |
| Document parsing (.docx/.pdf) | `unstructured.partition.auto.partition` is mocked for the element-joining unit test; real parsing of real sample files is exercised by `live`-marked tests |
| LLM (unit tests) | `FakeToolCallingChatModel` + `scripted_tool_call(name, args)` (`tests/fakes.py`) — the LangChain analog of pydantic-ai's `TestModel(custom_output_args=...)`; scripted per test since there's no post-hoc `.override()` hook |
| LLM (`provider="test"`) | `AutoStructuredFakeChatModel` (`api/test_provider.py`) — synthesizes valid structured output for any schema, so the shipped offline mode (not just unit tests) works end-to-end |
| Streamlit frontend | `streamlit.testing.v1.AppTest` drives the real script with `requests.Session` mocked at the `api_client` import site — no browser needed |

Coverage gate: `pytest`'s `addopts` enforces `--cov-fail-under=95` (actual is
~99%); `poc/*` is excluded (it's a manual, credential-requiring smoke test by
its own docstring).

## Out of scope for this port

`experiments/` and `notebooks/week4_retrieval_optimization.py` from the
original weren't ported. (The React frontend was replaced, not ported — this
version's frontend is the new Streamlit UI in `ui/`, not a port of the
original's React app.)
