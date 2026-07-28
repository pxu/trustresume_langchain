# Syncing from the original `trustresume` repo

This repo is a from-scratch port of
[`trustresume`](https://github.com/pxu/trustresume)
(`/Users/joe.xu/repo/trustresume` locally) onto LangChain + LangGraph +
ChromaDB. The original keeps evolving (course submissions, bug fixes, new
features); this doc is the procedure for pulling those changes in here
without re-deriving the whole port from scratch each time.

## Last synced commit

```
7efa36164488274843d9bac2dac6735914bf8080  (dated 2026-07-26)
```

This is the exact commit the initial port (2026-07-27) was read from. **Update
this line to the new HEAD SHA at the end of every sync.**

## Sync procedure

1. **Find what changed** in the original since the last sync:
   ```bash
   cd /Users/joe.xu/repo/trustresume
   git log --oneline <LAST_SYNCED_SHA>..HEAD
   git diff --stat <LAST_SYNCED_SHA>..HEAD
   ```
2. **Categorize each changed file** using the module map below — most files
   fall into "copy verbatim" (no translation work) or "re-apply conceptually"
   (the underlying behavior change matters, not the literal diff, because
   the implementation is already framework-specific).
3. **Apply the change**, module by module, verifying with `pytest` + `mypy
   src` + `ruff check .` after each one — same discipline as the original
   port (see git history / the original conversation: M1→M7, one module at a
   time, never batching).
4. **Update "Last synced commit"** above to the new HEAD SHA once everything
   passes.

Do not diff-and-`cp` blindly for anything in the "re-apply conceptually" rows
below — those files were rewritten onto a different stack, so a literal patch
won't apply and would reintroduce pydantic-ai/Qdrant-specific code. Read the
original's diff to understand *what changed behaviorally*, then make the
equivalent change in this repo's already-translated implementation.

## Module translation map

| Original path | Port path | Sync rule |
|---|---|---|
| `src/trustresume/models/*` | same | **Copy verbatim.** Pure pydantic, no framework dependency. |
| `src/trustresume/storage/*` | same | **Copy verbatim.** SQLite-only, not part of the stack swap. |
| `src/trustresume/trust_verification/*` | same | **Copy verbatim.** Pure prompt/formatting strings, framework-agnostic. |
| `src/trustresume/evaluation/*` | same | **Copy verbatim.** No LLM/framework dependency at all. |
| `src/trustresume/ingestion/parser.py`, `chunker.py` | same | **Copy verbatim.** Pure functions. |
| `src/trustresume/ingestion/service.py`, `__init__.py` | same | **Copy verbatim**, then swap any `QdrantVectorStore` type reference for `ChromaVectorStore` — the rollback/write-then-upsert contract itself doesn't change. |
| `src/trustresume/retrieval/query.py` | same | **Copy verbatim.** Pure string-building. |
| `src/trustresume/retrieval/embedder.py`, `vector_store.py` | same | **Re-apply conceptually.** Qdrant-specific; port onto `FastEmbedEmbeddings`/`ChromaVectorStore` (see ADR-0001). If the original adds a new field to what's stored/searched, add the equivalent Chroma metadata field and read-side reconstruction. |
| `src/trustresume/agents/*` | same | **Re-apply conceptually.** pydantic-ai `Agent`-specific; port onto `chat_model.with_structured_output(Schema)` (see `agents/base.py`, CLAUDE.md). Preserve any force-override post-processing (e.g. `raw_text`, `iteration`) exactly. |
| `src/trustresume/orchestration/orchestrator.py` | same | **Re-apply conceptually.** Hand-rolled loop → LangGraph `StateGraph` (ADR-0003). Re-derive the equivalent node/edge/conditional-edge change; do not copy the `while`-loop diff directly. |
| `src/trustresume/orchestration/feedback.py`, `candidate_profile_service.py` | same | **Copy verbatim.** No framework dependency. |
| `src/trustresume/api/model_factory.py` | same | **Copy the `LLMConfig` dataclass/precedence logic verbatim**; re-apply provider-dispatch changes onto the `langchain_aws`/`langchain_openai`/`langchain_google_genai` branches in `build_model`. |
| `src/trustresume/api/app_service.py`, `schemas.py`, `server.py` | same | **Copy verbatim**, swapping only `QdrantVectorStore`/pydantic-ai `Model` type references for `ChromaVectorStore`/`BaseChatModel`. |
| `src/trustresume/poc/llm_smoke_test.py` | same | **Re-apply conceptually** onto `langchain.agents.create_agent` — pydantic-ai/CodeMode-specific otherwise. |
| `tests/unit/test_models_*.py`, `test_storage.py`, `test_trust_verification.py`, `test_evaluation.py` | same | **Copy verbatim.** |
| `tests/unit/test_retrieval.py`, `test_ingestion.py`, `test_agents.py`, `test_orchestrator.py`, `test_candidate_profile_service.py`, `test_model_factory.py`, `tests/integration/*` | same | **Re-apply conceptually** using the fixtures in `tests/fakes.py` (`FakeEmbeddings`, `FakeToolCallingChatModel`/`scripted_tool_call`) instead of `QdrantClient`/pydantic-ai `TestModel`. See "Test translation patterns" below. |
| `tests/fakes.py`, `tests/conftest.py` | — | No original equivalent for the tool-calling fake; these are net-new support code for this stack (see CLAUDE.md). |
| `docs/`, `frontend/`, `experiments/`, `notebooks/`, `ui/` | not ported | Out of scope for this port (backend-only decision). Skip changes here unless the scope decision is revisited. |

## Test translation patterns

- `QdrantClient(":memory:")` → `chromadb.EphemeralClient()`, **with a unique
  `collection_name` per test** (`ChromaVectorStore(..., collection_name=f"test-{uuid.uuid4().hex}")`)
  — Chroma's ephemeral client caches storage by settings hash, so same-named
  collections leak state across "fresh" client instances in one process.
- `pydantic_ai.models.test.TestModel(custom_output_args={...})` +
  `agent._agent.override(model=tm)` → `tests/fakes.py`'s
  `scripted_tool_call(name, args)`, passed directly into the agent's
  constructor (no override hook exists in LangChain — script the fake at
  construction time instead). `name` must be the target Pydantic model's
  class name.
- A bare `TestModel()` (auto-synthesizes any schema) → for integration tests
  that run the *whole* pipeline, either script the exact call sequence with
  content engineered to pass the gate (see `tests/integration/test_app_service.py`'s
  `_FULL_GENERATION`), or use `trustresume.api.test_provider.AutoStructuredFakeChatModel`
  directly if the original test didn't care about exact scores.
- `FakeEmbedder` (single `embed()` method) → `tests/fakes.py`'s
  `FakeEmbeddings` (`embed_documents`/`embed_query` — LangChain's
  `Embeddings` ABC shape).

## Known, intentional deviations (not sync bugs)

These are permanent differences from the original — don't try to "fix" them
back into alignment during a sync:

- `api/test_provider.py`'s `AutoStructuredFakeChatModel` is new code with no
  original equivalent (LangChain's fakes can't auto-synthesize structured
  output the way pydantic-ai's `TestModel()` did).
- Because that synthesizer produces an empty `claims` list for the Trust
  Harness schema, `TRUSTRESUME_LLM_PROVIDER=test` always hits the iteration
  cap (Trust score 0) rather than passing — deterministic, not a bug.
- `chunk_id` is used as the Chroma document id directly; the original's
  uuid5 point-id indirection (required by Qdrant) doesn't exist here.
- Env var `TRUSTRESUME_QDRANT_PATH` → `TRUSTRESUME_CHROMA_PATH`.
