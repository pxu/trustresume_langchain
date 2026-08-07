"""``python -m trustresume.evals`` — run the evaluation suites and print results.

The only module here that builds real dependencies (a real embedder, a real
Chroma collection, a real LLM), which is why it's excluded from the coverage
gate: exercising it meaningfully needs the very things the offline test suite
exists to avoid, the same reasoning behind ``poc/`` and the ``live`` marker.
Everything it calls is dependency-injected and unit-tested.

    # Retrieval only — needs no credentials (fastembed runs locally)
    python -m trustresume.evals --suite retrieval

    # Both suites against a real LLM
    TRUSTRESUME_LLM_PROVIDER=bedrock python -m trustresume.evals --suite all

    # Record the run as the new baseline to compare future changes against
    python -m trustresume.evals --suite all --save evals/baselines/latest.json

Note the trust suite is meaningless under ``TRUSTRESUME_LLM_PROVIDER=test``:
the offline fake synthesizes an empty claim list, which this harness scores as
UNSUPPORTED for every case (see ``trust_eval._dominant_status``). That is
correct behavior, not a bug — but it means the run tells you nothing about a
real model, so the CLI says so rather than printing a confusing 0.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from trustresume.agents import TrustHarnessAgent
from trustresume.api.model_factory import LLMConfig, build_model
from trustresume.evals.datasets import (
    load_corpus,
    load_retrieval_cases,
    load_trust_cases,
    validate_dataset,
)
from trustresume.evals.retrieval_eval import (
    DEFAULT_K,
    EVAL_USER_ID,
    evaluate_retrieval,
    ingest_corpus,
)
from trustresume.evals.trust_eval import evaluate_trust
from trustresume.ingestion import IngestionService
from trustresume.models import DocumentType
from trustresume.retrieval import ChromaVectorStore, FastEmbedEmbeddings, HybridRetriever
from trustresume.retrieval.embedder import DEFAULT_MODEL as EMBEDDING_MODEL
from trustresume.storage import (
    CandidateProfileRepository,
    ChunkRepository,
    DocumentRepository,
    JobDocumentRepository,
    UserRepository,
    connect,
    init_db,
)


def _run_retrieval(k: int) -> dict[str, Any]:
    """Ingest the labeled corpus into throwaway stores and score retrieval.

    In-memory SQLite and an ephemeral Chroma collection, both discarded when
    the process exits: an evaluation must never read or write the developer's
    real ``trustresume.db``/``chroma_data``, or its numbers would depend on
    whatever happened to be ingested there.
    """
    import chromadb

    corpus = load_corpus()
    cases = load_retrieval_cases()
    validate_dataset(corpus, cases)

    connection = connect(":memory:")
    init_db(connection)
    documents = DocumentRepository(connection)
    chunks = ChunkRepository(connection)
    vectors = ChromaVectorStore(
        chromadb.EphemeralClient(),
        FastEmbedEmbeddings(),
        collection_name=f"evals-{uuid.uuid4().hex}",
    )
    ingestion = IngestionService(
        documents=documents,
        chunks=chunks,
        vector_store=vectors,
        candidate_profiles=CandidateProfileRepository(connection),
        job_documents=JobDocumentRepository(connection),
    )
    UserRepository(connection).create("Eval Harness", user_id=EVAL_USER_ID)

    def ingest(doc_id: str, text: str) -> str:
        # filename carries the label id; ingestion mints the real document id,
        # which ingest_corpus maps back for us.
        return ingestion.ingest_text(
            user_id=EVAL_USER_ID,
            filename=doc_id,
            text=text,
            document_type=DocumentType.OTHER,
        )

    labels = ingest_corpus(corpus, ingest)
    report = evaluate_retrieval(
        HybridRetriever(vectors, chunks), cases, corpus=corpus, k=k, labels=labels
    )
    connection.close()

    print(f"\n=== Retrieval (k={report.metrics.k}, {report.metrics.queries} queries) ===")
    print(f"  embedder     {EMBEDDING_MODEL}")
    print(f"  recall@k     {report.metrics.recall_at_k:.3f}")
    print(f"  precision@k  {report.metrics.precision_at_k:.3f}")
    print(f"  MRR          {report.metrics.mrr:.3f}")
    print(f"  hit rate     {report.metrics.hit_rate:.3f}")
    if report.metrics.unanswerable_queries:
        print(
            f"  unanswerable {report.metrics.unanswerable_queries} queries, "
            f"{report.metrics.unanswerable_results:.1f} results returned on average "
            f"(excluded from P/MRR/hit rate; lower is better)"
        )
    for case in report.failing_cases():
        print(f"  MISS {case.query_id}: {case.query!r} -> missed {case.missed_doc_ids}")
        if case.note:
            print(f"       ({case.note})")
    # The embedding model, not the LLM: this suite makes no LLM call, and
    # recording the configured provider here would credit the numbers to a
    # model that had nothing to do with them.
    return {"embedder": EMBEDDING_MODEL, **report.metrics.model_dump()}


async def _run_trust(config: LLMConfig) -> dict[str, Any]:
    """Score the Trust Harness against the labeled claim set."""
    if config.provider == "test":
        print(
            "\n=== Trust: SKIPPED ===\n"
            "  The offline 'test' provider returns no claims, which scores as\n"
            "  UNSUPPORTED for every case. Set TRUSTRESUME_LLM_PROVIDER to a real\n"
            "  provider to get a meaningful number."
        )
        return {}

    cases = load_trust_cases()
    agent = TrustHarnessAgent(build_model(config, role="verifier"))
    report = await evaluate_trust(agent, cases)

    print(f"\n=== Trust Harness ({report.metrics.cases} labeled claims) ===")
    print(f"  accuracy   {report.metrics.accuracy:.3f}")
    print(f"  macro F1   {report.metrics.macro_f1:.3f}")
    for label in report.metrics.per_label:
        print(
            f"  {label.label:<20} P {label.precision:.2f}  R {label.recall:.2f}  "
            f"F1 {label.f1:.2f}  (n={label.support})"
        )
    dangerous = report.dangerous_errors
    print(f"  too-lenient errors: {len(dangerous)}  <- the ones that ship a fabrication")
    for case in dangerous:
        print(f"    {case.case_id}: expected {case.expected.value}, got {case.predicted.value}")
        print(f"      claim: {case.claim}")
    return report.metrics.model_dump()


async def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run TrustResume's offline evaluation suites.")
    parser.add_argument(
        "--suite", choices=("retrieval", "trust", "all"), default="all", help="Which suite to run."
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K, help="Retrieval cut-off (default: the agent's top_k)."
    )
    parser.add_argument("--save", type=Path, help="Write the metrics to this JSON file.")
    args = parser.parse_args(argv)

    config = LLMConfig.from_env()
    results: dict[str, Any] = {}
    if args.suite in ("retrieval", "all"):
        results["retrieval"] = _run_retrieval(args.k)
    if args.suite in ("trust", "all"):
        trust = await _run_trust(config)
        # Provider/model recorded only when a suite actually used one — the
        # retrieval suite runs entirely on fastembed.
        if trust:
            results["trust"] = {
                "provider": config.provider,
                "model": config.model_name("verifier"),
                **trust,
            }

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved metrics to {args.save}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m trustresume.evals``."""
    return asyncio.run(_main(argv))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
