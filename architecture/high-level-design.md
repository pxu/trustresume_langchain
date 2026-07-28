# TrustResume (LangChain/LangGraph/Chroma port) — High-Level Design

The system as ported: major components, how data flows through a generation,
and the decisions that shape the whole. This is a from-scratch reimplementation
of [`trustresume`](https://github.com/pxu/trustresume) (pydantic-ai + Qdrant)
on LangChain + LangGraph + ChromaDB, keeping the same overall architecture and
the SQLite storage layer unchanged. Read the original's `architecture/` docs
for the fuller design rationale (requirements, ADRs 0002/0004–0009 all still
apply here); this doc covers what's the same, what changed, and why.

## System overview

```
                            ┌─────────────────────────────┐
                            │  FastAPI backend (api/)      │
                            │  ─ server.py (routes)        │
                            │  ─ TrustResumeApp (facade)   │
                            └─────────────┬───────────────┘
                                          │ drives
                            ┌─────────────▼───────────────┐
                            │  Orchestrator (LangGraph)    │
                            │  StateGraph owns the graph   │
                            │  state + the quality loop     │
                            └──┬───┬───┬───┬───┬───────────┘
                sequences ─────┘   │   │   │   └───────────────┐
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
                    └──────┬───────┬──────┘              └────────────────────────┘
                    ┌──────▼──┐ ┌──▼─────────┐
                    │ SQLite  │ │  Chroma     │   hybrid store (ADR-0001)
                    │ storage │ │  retrieval  │   both keyed by user_id
                    └─────────┘ └─────────────┘
```

A sixth agent, `CandidateProfileAgent`, isn't pictured — it's job-independent
and cached, invoked through `CandidateProfileService` rather than directly by
the orchestrator, exactly as in the original.

## Components

| Layer | Package | What changed from the original |
|---|---|---|
| **HTTP** | `api/server.py` | Unchanged (routes, DTO translation) — operates on `WorkflowState`/the facade, not agent internals. |
| **Facade** | `api/app_service.py` | Collaborator types only: `ChromaVectorStore` instead of `QdrantVectorStore`, a LangChain `BaseChatModel` instead of a pydantic-ai `Model`. Construction order/logic unchanged. |
| **LLM provider factory** | `api/model_factory.py` | Same `LLMConfig` + precedence logic (env > llm.local.json > llm.json > default); provider dispatch retargeted to `langchain_aws`/`langchain_openai`/`langchain_google_genai`. The offline `"test"` provider is a new, more capable `AutoStructuredFakeChatModel` (`api/test_provider.py`) that synthesizes valid structured output for *any* bound schema, so `TRUSTRESUME_LLM_PROVIDER=test` can drive the full pipeline with no credentials (the original's pydantic-ai `TestModel` had this "synthesize anything" ability natively; LangChain's fakes don't, so it's rebuilt here). |
| **Control** | `orchestration/orchestrator.py` | Rebuilt as a LangGraph `StateGraph` (ADR-0003, superseding the original's ADR-0003). Public contract (constructor, `run()` signature, return type) unchanged. `feedback.py`, `candidate_profile_service.py` unchanged. |
| **Workers** | `agents/` | Each LLM-backed agent swaps a pydantic-ai `Agent` for `chat_model.with_structured_output(Schema)`; the exact post-processing quirks (force-preserving `raw_text`, force-stamping `iteration`, `TrustHarnessAgent`'s private `_ClaimExtraction` schema) are preserved. Deterministic agents (`EvidenceRetrievalAgent`, `ATSEvaluationAgent`) are unchanged apart from the vector-store type. |
| **Domain logic** | `trust_verification/`, `evaluation/` | Unchanged — pure string-building and scoring functions, no framework dependency. |
| **Write path** | `ingestion/` | Unchanged except the vector-store type; the write-then-upsert, roll-back-on-failure contract is identical. |
| **Data** | `storage/` (SQLite), `retrieval/` (Chroma) | `storage/` ported byte-for-byte. `retrieval/` rebuilt on Chroma (ADR-0001): `embedder.py` implements `langchain_core.embeddings.Embeddings` instead of a custom `Embedder` protocol; `vector_store.py`'s `ChromaVectorStore` keeps the same three-method surface (`upsert_chunks`/`search`/`delete_chunks`) as the original's `QdrantVectorStore`. `query.py` unchanged. |
| **Contracts** | `models/` | Unchanged — pure pydantic, no framework dependency anywhere in this package. |

## The generation data flow

Unchanged in shape from the original (`POST /api/generate`, with documents
already ingested):

```
1. job_description_agent.run(posting)          → JobDescription  (once)
2. candidate_profile_service.get_or_refresh(user_id) → CandidateProfile (once; usually a cache hit)
3. retrieval_agent.run(user_id, job)            → EvidenceSet     (once; Chroma, user-filtered)
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
split, and the exact iteration-counting semantics are unchanged.

## Testing model

Same offline-first philosophy as the original (NFR-5): the whole stack runs
without network access or credentials.

| Real dependency | Test double |
|---|---|
| SQLite file | `connect(":memory:")` (unchanged) |
| Chroma | `chromadb.EphemeralClient()` |
| Embedding model | `FakeEmbeddings` (`tests/fakes.py`) — same SHA-256 hashing as the original's `FakeEmbedder`, adapted to `Embeddings`'s two-method interface |
| LLM (unit tests) | `FakeToolCallingChatModel` + `scripted_tool_call(name, args)` (`tests/fakes.py`) — the LangChain analog of pydantic-ai's `TestModel(custom_output_args=...)`; scripted per test since there's no post-hoc `.override()` hook |
| LLM (`provider="test"`) | `AutoStructuredFakeChatModel` (`api/test_provider.py`) — synthesizes valid structured output for any schema, so the shipped offline mode (not just unit tests) works end-to-end |

## Out of scope for this port

The React frontend, `experiments/`, and `notebooks/week4_retrieval_optimization.py`
from the original weren't ported — this port focused on the backend
(models → storage → retrieval → ingestion → agents → orchestration → API).
