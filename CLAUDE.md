# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This is a from-scratch reimplementation of
[`trustresume`](https://github.com/pxu/trustresume) (the original MSAI-699
capstone: RAG + multi-agent resume generation on pydantic-ai + Qdrant) on
**LangChain + LangGraph + ChromaDB**. Scope so far: the Python backend only
(models → storage → retrieval → ingestion → agents → orchestration → API);
the React frontend, `experiments/`, and the Week 4 notebook were not ported.
See `docs/architecture/high-level-design.md` and `docs/architecture/decisions/` for what
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

Offline evaluation (ADR-0011 — *not* part of `pytest`; this measures the
system's quality, not its correctness):

```bash
python -m trustresume.evals --suite retrieval   # no credentials needed (~10s)
TRUSTRESUME_LLM_PROVIDER=bedrock python -m trustresume.evals --suite all
python -m trustresume.evals --suite all --save evals/baselines/latest.json
```

Run the retrieval suite before/after any change to chunking, the embedder, or
retrieval fusion, and compare against `evals/baselines/latest.json`. See
`evals/README.md` for how to read the numbers.

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
# TRUSTRESUME_USER_ID prefills the sidebar's user id (sent as X-User-Id);
# blank means the demo user. Switching it switches to a separate workspace.
```

Provider/model selection is resolved by `api/model_factory.py`'s `LLMConfig`:
`config/llm.json` (committed defaults) → gitignored `config/llm.local.json`
overlay → environment variables (highest precedence). Fields: `provider`
(`bedrock` | `openai` | `google` | `test`), `model`, `api_key`, `aws_profile`,
`aws_region`, `temperature` (pinned, default `0`), and `roles` — per-role
model/temperature overrides for `extraction` | `writer` | `verifier`
(ADR-0013). LLM pricing for cost reporting is separate:
`config/pricing.json` (ADR-0012). `TRUSTRESUME_OUTPUT_DIR` (default `output`,
empty string disables) controls where each run's browsable copy is written.
`TrustResumeApp(output_dir=...)` defaults to `None` — writing nothing — so the
test suite never touches the filesystem; only `build_default_app` sets it.

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
| `storage/` | SQLite repositories (users, documents, **jobs**, **job↔document links**, chunks, resumes, evaluations, candidate-profile cache). The original's repositories are ported byte-for-byte; added on top of them are `documents.content_hash` (+ `UNIQUE(user_id, content_hash)`), the `chunks_fts` FTS5 table and its sync triggers, the `jobs`/`job_documents` tables, and the export/rejection/usage columns on `generated_resumes`. |
| `retrieval/` | `FastEmbedEmbeddings` + `ChromaVectorStore` (vector search) + `HybridRetriever` (vector + SQLite-FTS5 keyword search, fused by RRF — ADR-0010) + `query.py` (job → query string). Every search is scoped by `user_id`, and optionally narrowed further by a `document_ids` allow-list (job scoping, below). |
| `ingestion/` | Parse (`unstructured`, any format) → clean → dedup-check (content hash) → chunk (LangChain) → write both stores (`IngestionService`); rolls back SQLite chunk rows if the Chroma upsert fails; a duplicate (same user, same cleaned-text hash) short-circuits before any of that. |
| `agents/` | Six pure input→output agents (five run every generation, `CandidateProfileAgent` is cached). LLM-backed ones wrap `chat_model.with_structured_output(Schema)`. |
| `orchestration/` | `Orchestrator` (LangGraph `StateGraph`, see Architecture below), `CandidateProfileService` (cache-check wrapper), `build_feedback` (deterministic rewrite instructions for the *next LLM pass*), `build_rejection_reason` (`rejection.py` — a human-readable "why this capped-out draft failed", persisted and displayed; deliberately not folded into `build_feedback`, different audience). |
| `trust_verification/` | Prompt/formatting helpers + report assembly for the Trust Harness — pure functions, no LLM/framework dependency. |
| `evaluation/` | ATS keyword-coverage scoring — pure functions. **Product logic**: scores one résumé for the user at runtime. Not to be confused with `evals/`. |
| `evals/` | **Engineering logic**: scores the *system* against labeled ground truth, offline (ADR-0011). Retrieval metrics (recall@k/MRR) + Trust Harness classification accuracy, datasets in repo-root `evals/datasets/*.jsonl`, run via `python -m trustresume.evals`. Everything but `cli.py` is dependency-injected and unit-tested offline. |
| `telemetry.py` | `UsageTracker` (a LangChain callback capturing tokens/calls per model) + config-driven pricing → `RunUsage` (ADR-0012). A callback is the only place the raw `AIMessage` is still visible, since `with_structured_output` consumes it. |
| `export/` | `render_markdown` / `render_pdf` — pure `ResumeDraft` → bytes/str renderers, no dependency beyond `models` (+ `fpdf2` for PDF). `artifacts.py`'s `write_run_artifacts` additionally mirrors each run to a browsable directory (`output/<user_id>/<ts>-<job-slug>-<id>/` with `resume.md`/`resume.pdf`/`evaluation.md`/`evaluation.json`/`job.md`) — a convenience view, never a source of truth: `_persist` swallows `OSError` so a full disk can't fail a generation. `render_pdf` uses fpdf2's built-in Helvetica core font, which only encodes **Latin-1** — and `_persist` renders inline, so an unencodable character used to raise and destroy a whole generation *after* every LLM call was paid for. A real Bedrock run hit this on an **em dash**; curly quotes and ellipses do it too, and models emit all three constantly. `_to_latin1` now transliterates typographic punctuation to ASCII and degrades anything else to `?` with a warning, never raising. Scripts with no ASCII equivalent (CJK, Cyrillic) still render as `?` — fixing that means bundling a Unicode TTF. `render_markdown` is lossless and remains the faithful export. |
| `api/` | `TrustResumeApp` facade (documents · jobs CRUD · job-scoped upload/generate/list · resumes + exports · ad-hoc `search_evidence`), `model_factory.py` (provider-agnostic `LLMConfig`/`build_model`, temperature + role tiering), `test_provider.py` (`AutoStructuredFakeChatModel`), FastAPI `server.py` (same resource groups under `/api/...`, plus `/api/ping` and `/api/health`; every user-scoped route takes `CurrentUser`, resolved from `X-User-Id` — ADR-0014), wire DTOs in `schemas.py`. |
| `ui/` | Streamlit frontend (`streamlit_app.py`, four tabs: Documents · Generate · Jobs · Search) — a thin REST client (`api_client.py`'s `TrustResumeClient`) over the FastAPI backend; no imports from `trustresume.api`/`orchestration`/etc., so the dependency points one way (UI → HTTP → backend) and `streamlit` stays an optional (`ui`) extra. |
| `poc/` | Standalone LLM smoke test (`llm_smoke_test.py`), not part of the app; needs the `providers` extra for non-Bedrock; excluded from the coverage gate (see Testing model). |
| `logging_config.py` | Stdlib `logging` + a JSON `Formatter`; `configure_logging()` is called once by `server.py`'s `build_served_app` (never by library code, so importing anything else stays side-effect-free). |
| `prompting.py` | `wrap_untrusted(tag, text)` + `UNTRUSTED_INPUT_NOTICE` — the shared prompt-injection defense. Framework-free on purpose so both `agents/` and the deliberately zero-framework `trust_verification/` can use it (see Conventions). |

## Architecture

Same overall shape as the original: a single orchestrator drives
`JobDescriptionAgent → EvidenceRetrievalAgent → ResumeWriterAgent →
TrustHarnessAgent → ATSEvaluationAgent`, plus a sixth, cached
`CandidateProfileAgent` behind `CandidateProfileService`. What's different:

- **Orchestrator is a LangGraph `StateGraph`** (`orchestration/orchestrator.py`),
  not a hand-rolled `while` loop. `Orchestrator`'s constructor is unchanged
  from the original, and so is the original `run(*, user_id, job_posting,
  gate=None) -> WorkflowState` call — but `run` has since grown three
  keyword-only params for the persisted-job path: `job` (a pre-extracted
  `JobDescription`), `job_id`, and `document_ids`. **Exactly one of
  `job_posting` or `job` must be passed** — both or neither raises
  `ValueError`; `job_id`/`document_ids` are carried through for job-scoped
  retrieval and persistence but change no control flow. `run` is `async`
  (it `ainvoke`s the graph); `TrustResumeApp` wraps it in `asyncio.run`.
  The quality-loop iteration counting is subtle
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
- **A job is a persisted entity, and retrieval can be scoped to it** — added
  post-port, no equivalent in the original. `POST /api/jobs` runs the Job
  Description agent once and stores the extracted `JobDescription` as JSON
  (`jobs`), so `generate_for_job` **skips the Job Description agent entirely**
  and re-uses the stored extraction. Documents can be linked to a job
  (`job_documents`, via `POST /api/jobs/{id}/documents`). The eligibility rule
  is the easy thing to get backwards: `DocumentRepository.list_eligible_document_ids`
  returns the **generic pool** (documents linked to *no* job at all) ∪
  (documents linked to *this* job) — a document linked only to a *different*
  job is **not** eligible here. `TrustResumeApp.generate_for_job` resolves that
  id list and passes it into `Orchestrator.run(document_ids=...)`; the
  orchestrator has no `DocumentRepository` of its own and never resolves
  scoping itself. `document_ids=None` means "no extra narrowing" (the original,
  job-agnostic behavior); `document_ids=[]` short-circuits to an empty result.
- **Every persisted resume carries its own exports** — `TrustResumeApp._persist`
  renders PDF + Markdown bytes unconditionally (both `generate` paths) and
  stores them on `generated_resumes`, so `/api/resumes/{id}/pdf|markdown` is a
  straight read, not a re-render. A draft that failed the gate additionally
  gets `rejection_reason` (`build_rejection_reason`) and
  `improvement_suggestions` (`build_feedback`); both `None` for a passing
  draft. `_persist` sets `state.resume_id` so the caller can deep-link without
  a second lookup.
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
- **Every run is measured: tokens, cost, latency** (ADR-0012,
  `telemetry.py`). The orchestrator attaches one `UsageTracker` per run as a
  LangGraph callback, so it sees every agent's LLM call without any agent
  knowing it exists — necessary because `with_structured_output` consumes the
  raw `AIMessage`, so `usage_metadata` never reaches the call site. Per-node
  latency comes from `_timed()`, applied once at graph-construction time (a
  node added later is measured automatically); timings are a *list*, appended
  per execution, so "was the 3rd rewrite slower than the 1st" stays
  answerable. **An unpriced model reports `cost_usd=None`, never a partial
  total** — prices live in `config/pricing.json`, and a sum that quietly omits
  the most expensive model is worse than an honest "unknown". Both offline
  fakes emit `usage_metadata` + a `model_name` on purpose: without that, every
  token assertion in the suite would be vacuously zero.
- **Temperature is pinned (default 0) and models tier by role** (ADR-0013).
  Nothing used to pass `temperature`, so provider defaults applied — an
  unpinned sampling parameter underneath a *deterministic scoring* claim.
  Roles are `extraction` (Job Description + Candidate Profile) / `writer`
  (Resume Writer) / `verifier` (Trust Harness), defined by the job rather than
  the agent name. Tiering is opt-in: with no `roles` configured all three
  resolve to the same model, so existing callers and tests are unaffected.
- **The caller is resolved per request from `X-User-Id`** (ADR-0014,
  `server.py`'s module-level `resolve_user` + `CurrentUser`). Identity, not
  authentication — the point is that ADR-0001's isolation became *testable*:
  every route used to hardcode `DEMO_USER_ID`, so no test could show two users
  are actually isolated. **`resolve_user` must stay at module scope** and reach
  the facade via `request.app.state`: with `from __future__ import
  annotations`, a `CurrentUser` alias defined inside `create_app` is
  unresolvable from module globals and FastAPI silently degrades it to an
  unknown query parameter — *every route then 422s*. Caught only by running
  the app, not by mypy or ruff.
- **`evals/` measures the system; `evaluation/` measures a résumé** (ADR-0011).
  The names are one letter apart and the distinction is the whole point: the
  runtime quality gate and ATS score are product output, computed *from* what
  the agents reported, so a Trust Harness that rubber-stamps everything scores
  perfectly and fails invisibly. `evals/` scores both halves against labeled
  ground truth offline. Macro-F1 sits next to accuracy because the labels skew
  SUPPORTED (a harness that never says UNSUPPORTED looks ~80% accurate while
  failing at its only job), and too-lenient verdicts are counted separately
  because those are the errors that ship a fabrication.
- **CI/Docker are new, not part of the original port**: `.github/workflows/ci.yml`
  runs ruff/mypy/pytest on Python 3.11 and 3.13 from the locked `uv.lock`;
  `Dockerfile` is a multi-stage build with two runtime targets (`runtime` = the
  API, `ui` = Streamlit, `FROM runtime AS ui` so they share one venv);
  `docker-compose.yml` wires both together (`TRUSTRESUME_LLM_PROVIDER=test` by
  default, so `docker compose up` needs no credentials).

Read `docs/architecture/high-level-design.md` and the ADRs under
`docs/architecture/decisions/` before making non-trivial changes;
`docs/code-walkthrough.md` is a narrative learning guide to the whole
codebase (data flow, why each design decision, suggested reading order).

**ADR numbering is split across two repos** — `docs/architecture/decisions/` holds
ADR-0001, 0003, 0010, and 0011–0014 (this port's own new/changed decisions).
Docstrings here also cite ADR-0002 and ADR-0004…0009; those are the
**original** repo's ADRs, which carry over unchanged and were deliberately not
restated (see `docs/architecture/README.md`). Don't go looking for a missing file,
and don't renumber this repo's ADRs to close the gaps.

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
| LLM (unit tests) | `tests/fakes.py`'s `scripted_tool_call(name, args)` — builds a `FakeToolCallingChatModel` that returns one scripted tool call; `name` must match the target Pydantic model's class name. Messages come from `tool_call_message()`, which attaches `usage_metadata` + a `model_name` — omit those and every token/cost assertion silently passes on zeros (ADR-0012). |
| LLM (`provider="test"`, integration/live tests) | `api/test_provider.py`'s `AutoStructuredFakeChatModel` — also emits synthetic-but-plausible token counts (~4 chars/token, scaled to the real prompt). Its model id is deliberately absent from `config/pricing.json`, so offline runs report real tokens with an honest `cost_usd=None`. |
| Eval harness (`evals/`) | No LLM/embedder needed — the evaluators take injected `SupportsSearch`/`SupportsTrustRun` protocols, so `tests/unit/test_evals.py` drives them with scripted fakes. The *committed datasets* are validated by those tests too (unknown `doc_id`s, duplicate ids, full label coverage): a typo'd label silently depresses recall forever. Only `src/trustresume/evals/cli.py` builds real dependencies, and it's `omit`-ted from coverage like `poc/`. |
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
changes against data from an earlier version.** The most recent such change is
ADR-0012's usage columns on `generated_resumes` (`input_tokens`,
`output_tokens`, `llm_calls`, `cost_usd`, `duration_ms`). This is an accepted,
deliberate gap for a personal-project scale (see `docker-compose.yml`'s
comment on the same volume) — add real migrations only if this ever needs
to preserve data across a schema change in practice.

## Conventions

- No commits unless explicitly asked.
- **Never interpolate externally-sourced text into a prompt bare.** A job
  posting, an uploaded document, retrieved evidence — all of it goes through
  `prompting.py`'s `wrap_untrusted(tag, text)`, paired with
  `UNTRUSTED_INPUT_NOTICE` in the system prompt, so an injection attempt
  embedded in it reads as data rather than instructions. `prompting.py`
  imports nothing framework-specific on purpose: `trust_verification/` is
  deliberately dependency-free (ADR-0004, in the *original* repo's numbering)
  and must not pull in `langchain_core` just to reuse this.
- Each top-level package under `src/trustresume/` mirrors the original's
  module boundaries; when porting or extending a module, check the original
  repo (`/Users/joe.xu/repo/trustresume`) for the reference behavior before
  guessing at intent — this whole repo is a faithful port, not a redesign.
  The one intentional deviation is that framework-idiomatic LangChain
  constructs are preferred over hand-rolled equivalents where they're a clear
  win (e.g. `ingestion/chunker.py`'s `RecursiveCharacterTextSplitter`) — this
  is a LangChain *learning* project, so don't "restore" such code to match the
  original's hand-rolled version.
