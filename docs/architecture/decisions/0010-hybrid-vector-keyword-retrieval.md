# ADR-0010: Hybrid (vector + keyword) retrieval, fused by RRF

## Status
Accepted. New — no equivalent decision in the original repo. Retrieval there
(and in this port, until now) was Chroma/Qdrant vector search only.

Numbered 0010, not 0004: the original repo's own ADR-0004 (generation and
verification as separate LLM passes) is a decision this port already
restates as still applying (see `CLAUDE.md`) — this repo's ADR numbers are
independent of the original's, but picking a low number here would read as
colliding with/superseding that original ADR-0004, which it doesn't.

## Context
Pure vector search misses exact-term matches an embedding model treats as
merely "nearby": a job description naming "Kubernetes" specifically shouldn't
lose to a candidate's chunk about "Docker" just because they're semantically
close, and a resume that literally says "AWS Lambda" should win over one that
only says "serverless" when the job posting asks for Lambda by name. This
matters more than usual here because `FastEmbedEmbeddings`
(`BAAI/bge-small-en-v1.5`, 384-dim) is a small, general-purpose model, not one
tuned on résumé/job-posting vocabulary — exact technical terms (framework
names, certifications, tool names) are exactly the case it's weakest on.

Pure keyword search has the opposite failure mode (misses paraphrases with no
shared vocabulary), so production RAG systems standardly combine both rather
than picking one. SQLite already holds every chunk's full text (`chunks.text`,
written by `IngestionService` alongside the Chroma vector) — the keyword half
didn't need a new data store, only an index on data already there.

## Decision
Add a keyword search path over the existing SQLite `chunks` table via FTS5,
and fuse it with the existing Chroma vector search via **Reciprocal Rank
Fusion (RRF)**, not by combining raw scores.

- **`chunks_fts`** (`storage/schema.py`) is an FTS5 *external-content* virtual
  table (`content='chunks', content_rowid='rowid'`) — it indexes
  `chunks.text` without duplicating it, kept in sync by `AFTER INSERT`/
  `AFTER DELETE` triggers on `chunks`. `ChunkRepository.add`/
  `delete_for_document` don't need to know the index exists at all.
- **`ChunkRepository.search_keywords`** (`storage/repositories.py`) runs the
  FTS5 `MATCH` query, ranked by SQLite's built-in `bm25()`. A free-text query
  is sanitized into safe FTS5 syntax first (`_to_fts5_query`: tokenize to
  bare alphanumeric words, quote each as its own phrase, OR-join) — FTS5's
  query syntax treats `-`/`"`/`*`/`:`/`/`/parentheses specially, so a raw job
  description (which reliably contains several of those) would otherwise
  raise a syntax error on `MATCH`.
- **`HybridRetriever`** (`retrieval/hybrid.py`) queries both
  `ChromaVectorStore.search` and `ChunkRepository.search_keywords` (each
  asked for more candidates than the final `limit`, so fusion has real
  overlap to work with), then fuses by RRF: each chunk's score is
  `1 / (k + rank)` summed across every list it appears in (`k=60`, the
  original RRF paper's value). **RRF fuses on rank position, not on raw
  score** — Chroma's cosine similarity (`[0, 1]`, higher is better) and
  SQLite's `bm25()` (unbounded, sign-flipped, lower is better) live on
  incomparable scales, so averaging or weighting them directly would be
  combining numbers that don't mean the same thing. Rank position is the one
  thing both sources produce in a comparable form.
- `HybridRetriever.search(user_id, query, limit) -> EvidenceSet` matches
  `ChromaVectorStore.search`'s exact shape, so `EvidenceRetrievalAgent` (which
  now type-hints its dependency as a `Protocol`, not `ChromaVectorStore`
  specifically) and `TrustResumeApp.search_evidence` needed no shape changes
  — only which object they're constructed with.

## Consequences
- Every chunk write still touches exactly the same two systems (Chroma,
  SQLite) as before — FTS5 is not a third store, just an index the existing
  SQLite file also carries. Ingestion's write-then-upsert/rollback contract
  (`IngestionService.ingest_text`) is unaffected.
- A chunk found by only one source (vector *or* keyword) still surfaces if it
  ranked well there; a chunk found by both — even at different ranks in
  each — outranks one found by only one, which is the property that makes
  hybrid retrieval an improvement over either alone rather than just "two
  searches concatenated."
- `search_keywords`'s BM25 score/rank isn't reused for anything after
  fusion — `EvidenceChunk.score` from the keyword path is left `None`,
  since it isn't on the same scale as Chroma's similarity and RRF never
  needs the raw value, only the rank it produced during fusion.
- Query sanitization (`_to_fts5_query`) means the keyword side never sees the
  literal query string a user or job description provided — only its
  alphanumeric tokens, OR-joined. This is a deliberate simplification (no
  phrase-adjacency matching, no field weighting) traded for never raising on
  arbitrary input; revisit if keyword precision needs to improve further
  (e.g. weighting multi-word technical terms as phrases).
