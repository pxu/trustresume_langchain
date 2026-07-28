# TrustResume (LangChain / LangGraph / ChromaDB port)

Evidence-based, ATS-friendly resume generation using RAG, multi-agent AI, and
trust verification — an MSAI-699 capstone project.

This is a from-scratch reimplementation of
[`trustresume`](https://github.com/pxu/trustresume) (the original pydantic-ai +
Qdrant MVP) on **LangChain + LangGraph + ChromaDB**, keeping the SQLite storage
layer and the overall architecture (five/six agents, an orchestrated quality
loop, trust verification decoupled from generation) unchanged. See
`architecture/` for the living design and the original repo's ADR-0003, which
anticipated this migration ("migrating to LangGraph later is an isolated
change").

## Status

Porting in progress, module by module (M1–M7), each verified with
`pytest`/`mypy`/`ruff` before moving to the next. See `architecture/` for the
current module map.

## Tech stack

LangChain (`with_structured_output` agents) · LangGraph (orchestrator + quality
loop) · ChromaDB (vector storage) · SQLite (structured storage, unchanged from
the original) · FastAPI · FastEmbed (embeddings) · AWS Bedrock / OpenAI /
Google — configurable LLM provider.

## Getting started

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev,providers]"

pytest          # offline unit + integration tests
ruff check .    # lint
mypy src        # type-check
```
