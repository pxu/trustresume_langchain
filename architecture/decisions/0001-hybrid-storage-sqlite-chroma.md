# ADR-0001: Hybrid storage — SQLite for structured data, Chroma for semantic retrieval

## Status
Accepted. Restates the original repo's ADR-0001 with Qdrant replaced by
ChromaDB; the underlying decision (two stores, one for structured records and
one for embedded semantic search, joined by `user_id` + `chunk_id`) is
unchanged.

## Context
TrustResume needs to store two different kinds of data: structured records
(users, uploaded-document metadata, generated resumes, ATS/Trust scores) and
unstructured career evidence (resumes, project reports, STAR stories,
certifications) that must be searched semantically against a job description.

## Decision
Use SQLite for structured, relational data (ported byte-for-byte from the
original — it isn't part of the LangChain/Chroma stack swap) and Chroma
(via `langchain_chroma.Chroma`) as a separate vector store for chunked,
embedded document content. Every record and every chunk carries a `user_id`
so retrieval can be scoped per user — enforced server-side via Chroma's
`filter={"user_id": ...}` metadata filter on every search, the direct analog
of the original's Qdrant `Filter(must=[FieldCondition(...)])`.

One simplification Chroma allowed over Qdrant: Chroma accepts arbitrary
string document ids directly, so `chunk_id` is used as the Chroma id itself —
no uuid5 point-id indirection needed (Qdrant required int/UUID ids).

## Consequences
- Two storage systems to keep in sync (a chunk's SQLite metadata row and its
  Chroma vector must agree) — `IngestionService`'s write-then-upsert,
  roll-back-on-failure sequencing is unchanged from the original.
- SQLite keeps the app dependency-light and file-based — no server to run for
  the structured half.
- Chroma's default `similarity_search_with_score` returns a **distance**
  (lower = more similar) rather than Qdrant's cosine **similarity** (higher =
  more similar); `ChromaVectorStore.search` converts `score = 1 - distance`
  (collection configured with `hnsw:space: cosine`) so `EvidenceChunk.score`
  keeps its "higher = more relevant" meaning for the rest of the app.
- User isolation is still enforced by always filtering on `user_id`, not by
  separate databases per user — same caveat as the original: worth
  revisiting if the isolation guarantee ever needs to be stronger than "we
  remembered to filter."
