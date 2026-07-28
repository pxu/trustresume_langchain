# architecture/

The living technical design for this LangChain/LangGraph/ChromaDB port — as
opposed to the original [`trustresume`](https://github.com/pxu/trustresume)
repo's own `docs/design/`, which is the graded design *document* for that
project. This folder is for artifacts a contributor (or a future you) actually
needs to work on this code.

- `high-level-design.md` — components, the generation data flow, and how they
  map onto the original's architecture.
- `decisions/` — Architecture Decision Records (ADRs). Most of the original's
  ADRs (0002, 0004–0009) describe decisions that carry over unchanged here
  (single orchestrator owning state, Trust Harness as a separate LLM pass,
  the quality gate, configurable LLM provider, etc.) — only the two ADRs that
  actually changed for this port are restated: ADR-0001 (Chroma replaces
  Qdrant) and ADR-0003 (LangGraph replaces the hand-rolled orchestrator —
  this port *is* the "isolated change" the original's ADR-0003 anticipated).

## Reading order

New to the code? `high-level-design.md` → `decisions/0001-hybrid-storage-sqlite-chroma.md`
→ `decisions/0003-langgraph-orchestrator.md` → the relevant package's own
`README.md` under `src/trustresume/` (where one exists).
