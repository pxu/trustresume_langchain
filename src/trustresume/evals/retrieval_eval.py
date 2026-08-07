"""Retrieval evaluation: ingest a labeled corpus, run the queries, score them.

Answers "did changing the chunker / embedder / fusion make retrieval better or
worse", which nothing in the runtime pipeline can tell you: the quality gate
scores the *draft*, and a draft can look fine while retrieval quietly regressed
(the writer just grounds fewer claims). The walkthrough's own note that the
chunking switch "theoretically changes retrieval hits, and a production system
should measure it" is this module's reason for existing.

Dependencies are injected (a retriever, an ingest callable) rather than
constructed here, so the whole evaluator runs offline against fakes in the
test suite and against the real Chroma/FastEmbed stack from the CLI — the same
double-life ``TrustResumeApp`` already has.

Added post-port; no equivalent in the original.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from trustresume.agents.retrieval_agent import DEFAULT_TOP_K as RETRIEVAL_TOP_K
from trustresume.models import EvidenceSet

from .datasets import CorpusDocument, RetrievalCase, validate_dataset
from .metrics import RetrievalMetrics, aggregate_retrieval

logger = logging.getLogger(__name__)

#: Cut-off for the headline numbers. Deliberately equal to
#: ``EvidenceRetrievalAgent``'s default ``top_k`` (8), so the metric measures
#: exactly what a generation actually sees rather than an arbitrary ranking
#: depth. If that default changes, change this with it — a retrieval metric
#: measured at a depth the pipeline never uses describes nothing.
DEFAULT_K = RETRIEVAL_TOP_K

#: Evaluation runs under one synthetic user id — retrieval is user-scoped
#: (ADR-0001), so the harness needs an owner for its corpus, and a distinctive
#: one keeps eval data recognizable if it ever lands in a shared store.
EVAL_USER_ID = "eval-harness"


class SupportsSearch(Protocol):
    """The retrieval surface an evaluation needs.

    Structurally identical to what ``EvidenceRetrievalAgent`` depends on, so
    ``HybridRetriever``, ``ChromaVectorStore``, or a fake all satisfy it —
    which is what makes "score vector-only vs. hybrid on the same dataset" a
    one-argument change rather than a rewrite.
    """

    def search(
        self, *, user_id: str, query: str, limit: int = 5, document_ids: list[str] | None = None
    ) -> EvidenceSet: ...


class CaseResult(BaseModel):
    """One query's outcome, kept for per-case inspection, not just the mean."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    query: str
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    relevant_doc_ids: list[str] = Field(default_factory=list)
    note: str | None = None

    @property
    def missed_doc_ids(self) -> list[str]:
        """Relevant documents this query failed to retrieve — where to look first."""
        return [d for d in self.relevant_doc_ids if d not in self.retrieved_doc_ids]


class RetrievalReport(BaseModel):
    """Aggregate metrics plus every case, so a regression is diagnosable."""

    model_config = ConfigDict(extra="forbid")

    metrics: RetrievalMetrics
    cases: list[CaseResult] = Field(default_factory=list)

    def failing_cases(self) -> list[CaseResult]:
        """Cases that missed at least one labeled document."""
        return [case for case in self.cases if case.missed_doc_ids]


def ingest_corpus(
    corpus: list[CorpusDocument],
    ingest: Callable[[str, str], str],
) -> dict[str, str]:
    """Load the labeled corpus and return ``{stored document id: label id}``.

    Takes the ingest callable rather than an ``IngestionService`` so the
    evaluation exercises whatever path the caller wants measured — including,
    deliberately, the real parse → clean → chunk → embed chain, because
    chunking is one of the things being evaluated. Evaluating retrieval
    against pre-chunked text would hide exactly the regressions this exists
    to catch.

    The mapping exists because ingestion mints its own document ids; the
    dataset's ``doc_id`` labels are ours. Translating on the way *out* (see
    :func:`evaluate_retrieval`) keeps the ingestion API untouched — an
    evaluation harness shouldn't get to bend the production write path just
    to make its bookkeeping easier.
    """
    mapping: dict[str, str] = {}
    for document in corpus:
        stored_id = ingest(document.doc_id, document.text)
        if stored_id in mapping:
            # Ingestion deduplicates on the *cleaned-text* hash, so two corpus
            # entries differing only in whitespace return the same stored id.
            # Silently overwriting here would make one label unreachable:
            # every query relevant to it would score as a miss and recall
            # would be understated forever — the same class of silent metric
            # corruption validate_dataset exists to prevent, just via content
            # rather than via ids.
            raise ValueError(
                f"corpus documents {mapping[stored_id]!r} and {document.doc_id!r} "
                "have identical content and deduplicated to one document; "
                "make them distinct or drop one"
            )
        mapping[stored_id] = document.doc_id
    logger.info("eval corpus ingested", extra={"documents": len(corpus)})
    return mapping


def _doc_ids_in_rank_order(evidence: EvidenceSet, labels: dict[str, str]) -> list[str]:
    """Retrieved document ids, best-first, deduplicated, mapped to label ids.

    Chunk-level hits are collapsed to document level (keeping each document's
    *best* rank) because relevance is labeled per document: two chunks of the
    same résumé are one retrieval success, not two, and counting them twice
    would inflate precision for a system that returns near-duplicate chunks.
    """
    ordered: list[str] = []
    for chunk in evidence.chunks:
        doc_id = labels.get(chunk.document_id, chunk.document_id)
        if doc_id not in ordered:
            ordered.append(doc_id)
    return ordered


def evaluate_retrieval(
    retriever: SupportsSearch,
    cases: list[RetrievalCase],
    *,
    corpus: list[CorpusDocument] | None = None,
    k: int = DEFAULT_K,
    user_id: str = EVAL_USER_ID,
    labels: dict[str, str] | None = None,
) -> RetrievalReport:
    """Run every labeled query through ``retriever`` and score the rankings.

    ``corpus``, when given, is only used to validate the labels before doing
    any work — a mislabeled ``doc_id`` should fail the run, not silently
    depress recall (see :func:`~.datasets.validate_dataset`).

    ``labels`` is :func:`ingest_corpus`'s ``{stored id: label id}`` mapping.
    Omit it when the retriever already returns label ids (a fake in a test).
    """
    if corpus is not None:
        validate_dataset(corpus, cases)

    results: list[CaseResult] = []
    for case in cases:
        evidence = retriever.search(user_id=user_id, query=case.query, limit=k)
        results.append(
            CaseResult(
                query_id=case.query_id,
                query=case.query,
                retrieved_doc_ids=_doc_ids_in_rank_order(evidence, labels or {}),
                relevant_doc_ids=case.relevant_doc_ids,
                note=case.note,
            )
        )

    metrics = aggregate_retrieval(
        [(case.retrieved_doc_ids, case.relevant_doc_ids) for case in results], k
    )
    logger.info(
        "retrieval evaluation finished",
        extra={
            "queries": metrics.queries,
            "recall_at_k": round(metrics.recall_at_k, 3),
            "mrr": round(metrics.mrr, 3),
        },
    )
    return RetrievalReport(metrics=metrics, cases=results)
