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
`uv.lock` is committed — install *from* it (`--locked`) rather than
re-resolving, so local/CI/Docker all land on the exact same dependency
versions.

```bash
uv venv --python 3.13
source .venv/bin/activate
uv sync --locked --extra dev --extra providers --extra ui   # exact locked versions
# or, to re-resolve after touching pyproject.toml's dependencies:
uv pip install -e ".[dev,providers,ui]"    # then `uv lock` to update uv.lock

pytest                          # offline unit + integration tests; coverage gate baked into addopts (--cov-fail-under=95)
pytest -m live                  # + live tests that boot a real uvicorn server over HTTP, or hit a real embedding model
pytest tests/unit/test_foo.py::test_bar   # single test
pytest -k "expression"          # tests matching an expression
ruff check .                    # lint
ruff format --check src tests   # format check (CI scope — excludes docs/*.md's embedded code fences)
mypy src                        # type-check (strict)
```

CI (`.github/workflows/ci.yml`) runs exactly this sequence (lint → format
check → mypy → pytest) on Python 3.11 and 3.13, installing from `uv.lock`.

Run the API:

```bash
# Bedrock (default) — needs AWS creds for profile twdc-bedrock-central
uvicorn trustresume.api.server:build_served_app --factory --reload --port 8000

# Offline, no credentials needed at all:
TRUSTRESUME_LLM_PROVIDER=test uvicorn trustresume.api.server:build_served_app --factory --port 8000

# Or via Docker Compose (api + streamlit ui, offline provider by default):
docker compose up --build
```

Run the Streamlit frontend (needs the API already running; `ui` extra):

```bash
TRUSTRESUME_API_URL=http://localhost:8000 streamlit run src/trustresume/ui/streamlit_app.py
```

Provider/model selection is resolved by `api/model_factory.py`'s `LLMConfig`:
`config/llm.json` (committed defaults) → gitignored `config/llm.local.json`
overlay → environment variables (highest precedence). Fields: `provider`
(`bedrock` | `openai` | `google` | `test`), `model`, `api_key`, `aws_profile`,
`aws_region`.

`scripts/manual_rag_test.py` is a real-Bedrock, real-fastembed, real-Chroma/
SQLite end-to-end smoke test — ingests the two sample résumés in
`data/sample_documents/` and generates against a job posting (default
`data/sample_job_descriptions/Sample_Job_Description.docx`, or pass a path as
`argv[1]`). Uses isolated temp DB/Chroma paths, never the repo's default
`trustresume.db`/`chroma_data`. Not part of the test suite (no assertions,
just prints the resulting `WorkflowState` as JSON) — for manually eyeballing
a real generation, not CI.

## Package map (`src/trustresume/`)

| Package | Responsibility |
|---|---|
| `models/` | Shared Pydantic schemas (`extra="forbid"`), no framework dependency. Everything else imports from here. |
| `storage/` | SQLite repositories (users, documents, chunks, resumes, evaluations, candidate-profile cache) — ported unchanged from the original. |
| `retrieval/` | `FastEmbedEmbeddings` + `ChromaVectorStore` (vector search) + `HybridRetriever` (vector + SQLite-FTS5 keyword search, fused by RRF — ADR-0010), always scoped by `user_id`. |
| `ingestion/` | Parse (`unstructured`, any format) → clean → dedup-check (content hash) → chunk (LangChain) → write both stores (`IngestionService`); rolls back SQLite chunk rows if the Chroma upsert fails; a duplicate (same user, same cleaned-text hash) short-circuits before any of that. |
| `agents/` | Six pure input→output agents (five run every generation, `CandidateProfileAgent` is cached). LLM-backed ones wrap `chat_model.with_structured_output(Schema)`. |
| `orchestration/` | `Orchestrator` (LangGraph `StateGraph`, see Architecture below), `CandidateProfileService` (cache-check wrapper), `build_feedback` (deterministic rewrite instructions). |
| `trust_verification/` | Prompt/formatting helpers + report assembly for the Trust Harness — pure functions, no LLM/framework dependency. |
| `evaluation/` | ATS keyword-coverage scoring — pure functions. |
| `api/` | `TrustResumeApp` facade (also exposes `add_document_bytes`/`search_evidence`), `model_factory.py` (provider-agnostic `LLMConfig`/`build_model`), `test_provider.py` (`AutoStructuredFakeChatModel`), FastAPI `server.py` (routes: `/api/documents`, `/api/documents/upload`, `/api/search`, `/api/generate`, `/api/ping`), wire DTOs in `schemas.py`. |
| `ui/` | Streamlit frontend (`streamlit_app.py`) — a thin REST client (`api_client.py`'s `TrustResumeClient`) over the FastAPI backend; no imports from `trustresume.api`/`orchestration`/etc., so the dependency points one way (UI → HTTP → backend) and `streamlit` stays an optional (`ui`) extra. |
| `poc/` | Standalone LLM smoke test (`llm_smoke_test.py`), not part of the app; needs the `providers` extra for non-Bedrock; excluded from the coverage gate (see Testing model). |
| `logging_config.py` | Stdlib `logging` + a JSON `Formatter`; `configure_logging()` is called once by `server.py`'s `build_served_app` (never by library code, so importing anything else stays side-effect-free). |

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
- **`ResumeWriterAgent` binds a lenient private schema (`_DraftExtraction`),
  not `ResumeDraft` directly** — a real failure against Bedrock Claude: the
  model sometimes emits a section with an empty `heading` (a stray bullet
  grouped under no real heading), which fails `ResumeDraft`'s `min_length=1`
  and crashes structured-output parsing, losing the whole draft. The agent
  binds `_DraftExtraction` (headings may be empty), then `_to_resume_draft`
  relabels empty headings to `"Additional Information"` rather than dropping
  the bullets. Same "LLM emits, code cleans up" split `TrustHarnessAgent`
  already uses for scoring. See
  `test_agents.py::test_resumeAgent_emptySectionHeading_relabeledNotDropped`.
- **Retrieval runs on Chroma** (`retrieval/vector_store.py`'s
  `ChromaVectorStore`), not Qdrant. `chunk_id` is used as the Chroma document
  id directly (no uuid5 indirection needed, unlike Qdrant). The `user_id`
  metadata filter on every `search()` call is the isolation boundary
  (ADR-0001) — never drop it or filter client-side. Chroma returns a
  *distance* (lower = more similar); `ChromaVectorStore.search` converts to
  `score = 1 - distance` so higher still means more relevant.
- **Retrieval is hybrid (vector + keyword), not vector-only** (ADR-0010,
  `retrieval/hybrid.py`'s `HybridRetriever`) — added post-port, no equivalent
  in the original at all. `ChunkRepository.search_keywords`
  (`storage/repositories.py`) runs BM25 over a new `chunks_fts` FTS5 virtual
  table (`storage/schema.py`, external-content, kept in sync by triggers on
  `chunks`); `_to_fts5_query` sanitizes free text into safe FTS5 syntax first
  (FTS5's `MATCH` raises on raw `-`/`"`/`*`/`:`/`/`/parens, which a job
  description reliably contains). `HybridRetriever` fuses Chroma's vector hits
  with the keyword hits via **Reciprocal Rank Fusion** (`1/(k+rank)` summed
  per source, `k=60`) — never by combining raw scores, since cosine
  similarity `[0,1]` and BM25's unbounded rank aren't on comparable scales.
  Matches `ChromaVectorStore.search`'s exact shape
  (`search(user_id, query, limit) -> EvidenceSet`), so `EvidenceRetrievalAgent`
  (now typed against a `Protocol`, not `ChromaVectorStore` specifically) and
  `TrustResumeApp.search_evidence` needed no call-site changes.
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
- **Chunking uses LangChain's `RecursiveCharacterTextSplitter`**
  (`ingestion/chunker.py`), not the original's hand-rolled loop — a deliberate
  exception to the "faithful port" rule, since adopting the framework's
  standard splitter is the point of a LangChain learning port. `chunk_text`'s
  public signature (`max_chars`/`overlap`) is unchanged so callers are
  unaffected. Two subtleties: (1) `separators=["\n", " ", ""]`, *not* the
  default `["\n\n", ...]`, because `clean_text` collapses blank lines to a
  single `\n` (there is no `\n\n` in cleaned text); (2) unlike the original,
  overlap is now applied between *all* adjacent chunks, not only when
  hard-splitting an oversized paragraph — so chunk boundaries (and thus
  retrieval hits / scores) differ from the original by design. `clean_text`
  stays hand-rolled — it's whitespace normalization, not chunking.
- **Parsing (`.docx`/`.pdf`) goes through `unstructured`**
  (`ingestion/parser.py`'s `_parse_rich_bytes`), one shared entry point for
  every non-plain-text format instead of a function per library
  (python-docx for `.docx`, pypdf for `.pdf`) — added post-port; PDF support
  didn't exist at all until then. **`strategy="fast"` is deliberate, not the
  default** — `unstructured`'s default `"hi_res"` strategy loads a torch-based
  layout model and took ~49s to parse one résumé PDF in testing; `"fast"`
  (plain text extraction, no layout model) does the same job in ~3s, and this
  app only needs the text, not a layout-aware read. (`langchain-community`'s
  `PyPDFLoader`/`Docx2txtLoader` were considered and rejected — that package
  is explicitly deprecated/sunset upstream, so adopting it would mean
  standardizing on something already being replaced.)
- **Ingestion deduplicates by content hash** (`ingestion/service.py`'s
  `_content_hash` + `ingest_text`) — added post-port; no equivalent in the
  original. Hashes the *cleaned* text (post `clean_text`), not raw bytes, so
  two uploads differing only in whitespace/encoding still count as the same
  document. `DocumentRepository.find_by_content_hash` is checked before any
  chunk/embed/write; a hit returns the existing `document_id` as a no-op. The
  DB also enforces `UNIQUE(user_id, content_hash)` (`storage/schema.py`) as a
  second line of defense against the check-then-insert race (two concurrent
  ingests of the same content) — `ingest_text` catches the resulting
  `sqlite3.IntegrityError` and re-reads the winning row rather than
  propagating it.
- **Structured JSON logging** (`logging_config.py`) — every module logs via
  stdlib `logging.getLogger(__name__)` with `extra={...}` context (`user_id`,
  `iteration`, etc.), rendered as one JSON object per line by `JsonFormatter`.
  The orchestrator logs each node transition and the quality-gate routing
  decision; ingestion logs success/rollback; the API logs request/response
  outcomes. **Never pass `extra={"filename": ...}`** (or `module`/`name`/
  `lineno`/etc.) — those collide with `LogRecord`'s own reserved attributes
  and raise `KeyError` inside stdlib `logging` (a real bug caught only by
  actually booting the server, not by unit tests that construct `LogRecord`
  directly — see `test_logging_config.py`'s
  `test_loggerInfo_withExtra_doesNotRaiseOnReservedLookingKeys`).
- **`/api/search`** (`api/server.py` + `TrustResumeApp.search_evidence`) exposes
  the Evidence Retrieval agent's hybrid search standalone, for a caller (the
  Streamlit UI's Search tab) to inspect retrieval quality directly instead of
  only seeing its effect buried inside a full `/api/generate` run.
- **CI/Docker are new, not part of the original port**: `.github/workflows/ci.yml`
  runs ruff/mypy/pytest on Python 3.11 and 3.13 from the locked `uv.lock`;
  `Dockerfile` is a multi-stage build with two runtime targets (`runtime` = the
  API, `ui` = Streamlit, `FROM runtime AS ui` so they share one venv);
  `docker-compose.yml` wires both together (`TRUSTRESUME_LLM_PROVIDER=test` by
  default, so `docker compose up` needs no credentials).

Read `architecture/high-level-design.md` and the ADRs under
`architecture/decisions/` before making non-trivial changes;
`docs/code-walkthrough.md` is a narrative learning guide to the whole
codebase (data flow, why each design decision, suggested reading order).

## Testing model

Whole stack runs offline, no network/credentials, mirroring the original's
NFR-5:

| Real dependency | Test double |
|---|---|
| SQLite file | `connect(":memory:")` |
| Chroma | `chromadb.EphemeralClient()` — give each test/store a **unique collection name**; Chroma's ephemeral client caches its underlying storage by settings hash, so same-named collections leak state across "fresh" client instances within one process. `TrustResumeApp.__init__` exposes `chroma_collection_name` for exactly this — `TrustResumeApp(...)` defaults to the shared production name, tests pass a unique one. |
| SQLite FTS5 (keyword search) | No fake needed — `connect(":memory:")` has FTS5 built in, so `ChunkRepository.search_keywords`/`HybridRetriever` tests run against the real index. |
| Document parsing (`.docx`/`.pdf`) | `unstructured.partition.auto.partition` is mocked for the element-joining unit test (`test_parseBytes_richDocument_joinsPartitionedElements`); real parsing of real sample files is exercised by two `live`-marked tests, one per format. |
| Embedding model | `tests/fakes.py`'s `FakeEmbeddings`; the real `FastEmbedEmbeddings`'s lazy-load contract is unit-tested by mocking `fastembed.TextEmbedding`, and its actual output is exercised by a `live`-marked test (real model, may download on first run) |
| LLM (unit tests) | `tests/fakes.py`'s `scripted_tool_call(name, args)` — builds a `FakeToolCallingChatModel` that returns one scripted tool call; `name` must match the target Pydantic model's class name |
| LLM (`provider="test"`, integration/live tests) | `api/test_provider.py`'s `AutoStructuredFakeChatModel` |
| Streamlit frontend | `streamlit.testing.v1.AppTest` (`tests/unit/test_streamlit_app.py`) drives the real script (widgets, clicks, form submission) with `requests.Session` mocked at `trustresume.ui.api_client`'s import site — no browser, no real HTTP. `st.cache_resource.clear()` must run between tests (autouse fixture) or a later test reuses an earlier test's mocked client. A prior manual Playwright check caught a real bug this way: `streamlit run` executes the file with no package context, so a relative import (`from .api_client import ...`) crashed — fixed by using an absolute import. |

`tests/unit/` mirrors `src/trustresume/`'s package boundaries one-to-one;
`tests/integration/` exercises `TrustResumeApp` and the FastAPI app end to
end. `tests/integration/test_api_live.py` boots a real uvicorn subprocess and
is marked `live` (deselected by default — `pytest -m live` to run it).

**Coverage gate**: `pytest`'s `addopts` bakes in `--cov-fail-under=95` (actual
is ~99% at last count) — a real regression fails the run, not just a report.
`src/trustresume/poc/*` is `omit`-ted from coverage (`[tool.coverage.run]`):
it's throwaway live-provider validation code by its own module docstring,
so meaningfully testing it needs a real LLM call — the same reasoning behind
the `live` marker.

## Schema changes have no migration tooling

`storage/schema.py`'s `init_db` only ever runs `CREATE TABLE/INDEX IF NOT
EXISTS` — there is no `ALTER TABLE` migration path anywhere in this project.
A schema change (new table, new column) is invisible to `init_db` against a
database file created before that change; the first write that touches the
new column fails at runtime ("no column named ...") rather than at startup.
**Delete `trustresume.db`/`chroma_data` locally, or run `docker compose down
-v` to drop the `trustresume-data` volume, before running code with schema
changes against data from an earlier version.** This is an accepted,
deliberate gap for a personal-project scale (see `docker-compose.yml`'s
comment on the same volume) — add real migrations only if this ever needs
to preserve data across a schema change in practice.

## Conventions

- No commits unless explicitly asked.
- Each top-level package under `src/trustresume/` mirrors the original's
  module boundaries; when porting or extending a module, check the original
  repo (`/Users/joe.xu/repo/trustresume`) for the reference behavior before
  guessing at intent — this whole repo is a faithful port, not a redesign.
  The one intentional deviation is that framework-idiomatic LangChain
  constructs are preferred over hand-rolled equivalents where they're a clear
  win (e.g. `ingestion/chunker.py`'s `RecursiveCharacterTextSplitter`) — this
  is a LangChain *learning* project, so don't "restore" such code to match the
  original's hand-rolled version.
