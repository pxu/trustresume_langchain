# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This is a from-scratch reimplementation of
[`trustresume`](https://github.com/pxu/trustresume) (the original MSAI-699
capstone: RAG + multi-agent resume generation on pydantic-ai + Qdrant) on
**LangChain + LangGraph + ChromaDB**. Scope so far: the Python backend only
(models → storage → retrieval → ingestion → agents → orchestration → API);
the React frontend, `experiments/`, and the Week 4 notebook were not ported.
See `architecture/high-level-design.md` and `architecture/decisions/` for what
changed and why (ADR-0001: Chroma replaces Qdrant; ADR-0003: LangGraph
replaces the hand-rolled orchestrator).

The original repo keeps evolving independently. **See `SYNC.md` for the
procedure to pull changes from `/Users/joe.xu/repo/trustresume` into this
repo** — it tracks the last-synced commit SHA and maps every original module
to either "copy verbatim" or "re-apply conceptually onto the LangChain/
LangGraph/Chroma equivalent." Read it before making changes prompted by "the
original repo added/changed X."

**Personal project, not Disney work.** This lives on the personal GitHub
account `pxu` (`git@github-pxu:pxu/trustresume_langchain.git`), with git
identity `pxu <xupengfei@gmail.com>` configured locally for this repo. Do not
apply Disney org conventions here (branch naming, Jira tickets, AI-Assisted-By
commit trailers, doc-sync tables) — treat it as an academic/portfolio project.
No commits unless explicitly asked.

## Commands

Python >= 3.11, `src` layout, managed with [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev,providers]"   # package + pytest/ruff/mypy + langchain-aws/openai/google-genai

pytest                          # offline unit + integration tests (live tests deselected)
pytest -m live                  # + live tests that boot a real uvicorn server over HTTP
pytest tests/unit/test_foo.py::test_bar   # single test
pytest -k "expression"          # tests matching an expression
ruff check .                    # lint
ruff format .                   # format
mypy src                        # type-check (strict)
```

Run the API:

```bash
# Bedrock (default) — needs AWS creds for profile twdc-bedrock-central
uvicorn trustresume.api.server:build_served_app --factory --reload --port 8000

# Offline, no credentials needed at all:
TRUSTRESUME_LLM_PROVIDER=test uvicorn trustresume.api.server:build_served_app --factory --port 8000
```

Provider/model selection is resolved by `api/model_factory.py`'s `LLMConfig`:
`config/llm.json` (committed defaults) → gitignored `config/llm.local.json`
overlay → environment variables (highest precedence). Fields: `provider`
(`bedrock` | `openai` | `google` | `test`), `model`, `api_key`, `aws_profile`,
`aws_region`.

## Package map (`src/trustresume/`)

| Package | Responsibility |
|---|---|
| `models/` | Shared Pydantic schemas (`extra="forbid"`), no framework dependency. Everything else imports from here. |
| `storage/` | SQLite repositories (users, documents, chunks, resumes, evaluations, candidate-profile cache) — ported unchanged from the original. |
| `retrieval/` | `FastEmbedEmbeddings` + `ChromaVectorStore` — embeds and semantically searches candidate evidence chunks, always scoped by `user_id`. |
| `ingestion/` | Parse → clean → chunk → write both stores (`IngestionService`); rolls back SQLite chunk rows if the Chroma upsert fails. |
| `agents/` | Six pure input→output agents (five run every generation, `CandidateProfileAgent` is cached). LLM-backed ones wrap `chat_model.with_structured_output(Schema)`. |
| `orchestration/` | `Orchestrator` (LangGraph `StateGraph`, see Architecture below), `CandidateProfileService` (cache-check wrapper), `build_feedback` (deterministic rewrite instructions). |
| `trust_verification/` | Prompt/formatting helpers + report assembly for the Trust Harness — pure functions, no LLM/framework dependency. |
| `evaluation/` | ATS keyword-coverage scoring — pure functions. |
| `api/` | `TrustResumeApp` facade, `model_factory.py` (provider-agnostic `LLMConfig`/`build_model`), `test_provider.py` (`AutoStructuredFakeChatModel`), FastAPI `server.py`, wire DTOs in `schemas.py`. |
| `poc/` | Standalone LLM smoke test (`llm_smoke_test.py`), not part of the app; needs the `providers` extra for non-Bedrock. |

## Architecture

Same overall shape as the original: a single orchestrator drives
`JobDescriptionAgent → EvidenceRetrievalAgent → ResumeWriterAgent →
TrustHarnessAgent → ATSEvaluationAgent`, plus a sixth, cached
`CandidateProfileAgent` behind `CandidateProfileService`. What's different:

- **Orchestrator is a LangGraph `StateGraph`** (`orchestration/orchestrator.py`),
  not a hand-rolled `while` loop — but `Orchestrator`'s constructor and
  `run(*, user_id, job_posting, gate=None) -> WorkflowState` are unchanged, so
  every caller is unaffected. The quality-loop iteration counting is subtle
  and load-bearing: `max_iterations=3` (the default `QualityGate`) yields
  **4** total drafts (iterations 0-3), not 3 — the conditional edge checks
  `iteration >= max_iterations` *before* `prepare_rewrite` increments it.
  See ADR-0003 and `test_orchestrator.py::test_orchestrator_failsToCap_stopsAndExportsRealScores`.
- **Agents use `chat_model.with_structured_output(Schema)`**
  (`agents/base.py`'s `ModelInput = BaseChatModel`), not a pydantic-ai
  `Agent`. Every agent constructor now *requires* its model — LangChain has
  no `.override(model=...)` hook, so tests inject a fake model directly at
  construction instead of swapping it post-hoc.
- **Retrieval runs on Chroma** (`retrieval/vector_store.py`'s
  `ChromaVectorStore`), not Qdrant. `chunk_id` is used as the Chroma document
  id directly (no uuid5 indirection needed, unlike Qdrant). The `user_id`
  metadata filter on every `search()` call is the isolation boundary
  (ADR-0001) — never drop it or filter client-side. Chroma returns a
  *distance* (lower = more similar); `ChromaVectorStore.search` converts to
  `score = 1 - distance` so higher still means more relevant.
- **The `"test"` LLM provider is `AutoStructuredFakeChatModel`**
  (`api/test_provider.py`), not a bare `GenericFakeChatModel` — it
  synthesizes a minimal valid instance of whatever schema an agent binds via
  `with_structured_output`, so `TRUSTRESUME_LLM_PROVIDER=test` can actually
  drive the full pipeline with no credentials (LangChain's built-in fakes
  don't support this; pydantic-ai's `TestModel` did, natively). Consequence:
  the synthesized `_ClaimExtraction.claims` is always empty, so Trust score
  is always 0 under `provider="test"` — the offline mode always hits the
  iteration cap rather than passing; this is deterministic, not a bug.
- **Storage is unchanged** (`storage/`, SQLite) — it isn't part of the stack
  being swapped, so it's ported byte-for-byte from the original.

Read `architecture/high-level-design.md` and the ADRs under
`architecture/decisions/` before making non-trivial changes.

## Testing model

Whole stack runs offline, no network/credentials, mirroring the original's
NFR-5:

| Real dependency | Test double |
|---|---|
| SQLite file | `connect(":memory:")` |
| Chroma | `chromadb.EphemeralClient()` — give each test/store a **unique collection name**; Chroma's ephemeral client caches its underlying storage by settings hash, so same-named collections leak state across "fresh" client instances within one process |
| Embedding model | `tests/fakes.py`'s `FakeEmbeddings` |
| LLM (unit tests) | `tests/fakes.py`'s `scripted_tool_call(name, args)` — builds a `FakeToolCallingChatModel` that returns one scripted tool call; `name` must match the target Pydantic model's class name |
| LLM (`provider="test"`, integration/live tests) | `api/test_provider.py`'s `AutoStructuredFakeChatModel` |

`tests/unit/` mirrors `src/trustresume/`'s package boundaries one-to-one;
`tests/integration/` exercises `TrustResumeApp` and the FastAPI app end to
end. `tests/integration/test_api_live.py` boots a real uvicorn subprocess and
is marked `live` (deselected by default — `pytest -m live` to run it).

## Conventions

- No commits unless explicitly asked.
- Each top-level package under `src/trustresume/` mirrors the original's
  module boundaries; when porting or extending a module, check the original
  repo (`/Users/joe.xu/repo/trustresume`) for the reference behavior before
  guessing at intent — this whole repo is a faithful port, not a redesign.
