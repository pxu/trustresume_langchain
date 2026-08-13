# architecture/

The living technical design for this LangChain/LangGraph/ChromaDB port — as
opposed to the original [`trustresume`](https://github.com/pxu/trustresume)
repo's own `docs/design/`, which is the graded design *document* for that
project. This folder is for artifacts a contributor (or a future you) actually
needs to work on this code.

- `high-level-design.md` — components, the generation data flow, and how they
  map onto the original's architecture.
- `decisions/` — Architecture Decision Records (ADRs). Most of the original's
  ADRs (0002, 0004–0009 in *its* numbering) describe decisions that carry
  over unchanged here (single orchestrator owning state, Trust Harness as a
  separate LLM pass, the quality gate, configurable LLM provider, etc.) —
  those aren't restated. This repo's own ADRs are numbered independently and
  cover only what's actually new or changed for this port:

  | ADR | Decision |
  |---|---|
  | 0001 | Chroma replaces Qdrant (and `user_id` filtering as *the* isolation boundary) |
  | 0003 | LangGraph replaces the hand-rolled orchestrator — this port *is* the "isolated change" the original's ADR-0003 anticipated |
  | 0010 | Hybrid vector + keyword retrieval, fused by RRF — no equivalent in the original |
  | 0011 | An offline evaluation harness with labeled ground truth |
  | 0012 | Per-run token, cost, and latency accounting |
  | 0013 | Pinned sampling temperature; models tiered by agent role |
  | 0014 | Per-request user identity, making ADR-0001 demonstrable |
  | 0015 | Opt-in durable execution (LangGraph checkpointing), demonstrative |

## Reading order

New to the code? `high-level-design.md` → `decisions/0001-hybrid-storage-sqlite-chroma.md`
→ `decisions/0003-langgraph-orchestrator.md` → `decisions/0010-hybrid-vector-keyword-retrieval.md`
→ `docs/code-walkthrough.md` (one level up) for a narrative, example-driven
tour of the whole codebase.

ADRs 0011–0014 are the "can you tell whether this is working?" layer — read
them together, and alongside `evals/README.md`, once the pipeline itself makes
sense. ADR-0015 (durable execution) is optional reading: an opt-in,
demonstrative feature that doesn't affect the default pipeline at all.
