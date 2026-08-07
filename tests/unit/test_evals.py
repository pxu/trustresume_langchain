"""Unit tests for the evaluation harness: dataset loading and both evaluators.

Runs entirely offline against fakes — the harness's own logic (label mapping,
rank collapsing, verdict reduction) is what's under test here, not the quality
of any model. The committed datasets are also validated, since a broken
dataset silently corrupts every number the harness reports.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from trustresume.evals.datasets import (
    CorpusDocument,
    RetrievalCase,
    TrustCase,
    load_corpus,
    load_retrieval_cases,
    load_trust_cases,
    validate_dataset,
)
from trustresume.evals.retrieval_eval import evaluate_retrieval, ingest_corpus
from trustresume.evals.trust_eval import evaluate_trust
from trustresume.models import (
    ClaimStatus,
    EvidenceChunk,
    EvidenceSet,
    ResumeDraft,
    TrustReport,
    VerifiedClaim,
)


class FakeRetriever:
    """Returns a scripted ranking per query, shaped like the real retriever."""

    def __init__(self, rankings: dict[str, list[str]]) -> None:
        self._rankings = rankings

    def search(
        self, *, user_id: str, query: str, limit: int = 5, document_ids: list[str] | None = None
    ) -> EvidenceSet:
        doc_ids = self._rankings.get(query, [])[:limit]
        return EvidenceSet(
            user_id=user_id,
            query=query,
            chunks=[
                EvidenceChunk(
                    chunk_id=f"{doc_id}-0", document_id=doc_id, user_id=user_id, text="..."
                )
                for doc_id in doc_ids
            ],
        )


class FakeTrustAgent:
    """Returns a scripted verdict per claim text."""

    def __init__(self, verdicts: dict[str, list[ClaimStatus]]) -> None:
        self._verdicts = verdicts

    async def run(self, *, draft: ResumeDraft, evidence: EvidenceSet) -> TrustReport:
        claim_text = draft.sections[0].bullets[0]
        statuses = self._verdicts.get(claim_text, [])
        claims = [VerifiedClaim(text=claim_text, status=status) for status in statuses]
        return TrustReport(claims=claims, score=TrustReport.compute_score(claims))


# --- datasets --------------------------------------------------------------


def test_committedDatasets_loadAndValidate() -> None:
    """The shipped datasets must parse and reference only existing documents."""
    corpus = load_corpus()
    cases = load_retrieval_cases()
    trust_cases = load_trust_cases()

    validate_dataset(corpus, cases)  # raises if a label points at a missing doc
    assert len(corpus) >= 10
    assert len(cases) >= 8
    assert len(trust_cases) >= 10


def test_committedTrustDataset_coversEveryStatus() -> None:
    """A set with no UNSUPPORTED cases couldn't detect a rubber-stamping harness."""
    statuses = {case.expected_status for case in load_trust_cases()}
    assert statuses == set(ClaimStatus)


def test_loadJsonl_skipsCommentsAndBlankLines(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text('# a header comment\n\n{"doc_id": "d1", "text": "hello"}\n', encoding="utf-8")
    assert load_corpus(path) == [CorpusDocument(doc_id="d1", text="hello")]


def test_validateDataset_unknownDocId_raises() -> None:
    """A typo'd label would otherwise depress recall forever, invisibly."""
    corpus = [CorpusDocument(doc_id="d1", text="x")]
    cases = [RetrievalCase(query_id="q1", query="q", relevant_doc_ids=["d1", "typo"])]

    with pytest.raises(ValueError, match="unknown doc_id"):
        validate_dataset(corpus, cases)


def test_validateDataset_duplicateDocId_raises() -> None:
    corpus = [CorpusDocument(doc_id="d1", text="x"), CorpusDocument(doc_id="d1", text="y")]
    with pytest.raises(ValueError, match="duplicate doc_id"):
        validate_dataset(corpus, [])


# --- retrieval evaluation --------------------------------------------------


def test_evaluateRetrieval_scoresRankingsAgainstLabels() -> None:
    cases = [
        RetrievalCase(query_id="q1", query="python", relevant_doc_ids=["d1"]),
        RetrievalCase(query_id="q2", query="rust", relevant_doc_ids=["d2"]),
    ]
    retriever = FakeRetriever({"python": ["d1", "d9"], "rust": ["d8", "d7"]})

    report = evaluate_retrieval(retriever, cases, k=2)

    assert report.metrics.recall_at_k == 0.5
    assert [c.query_id for c in report.failing_cases()] == ["q2"]
    assert report.failing_cases()[0].missed_doc_ids == ["d2"]


def test_evaluateRetrieval_collapsesChunksOfOneDocumentToOneHit() -> None:
    """Three chunks of the same résumé are one retrieval success, not three."""
    cases = [RetrievalCase(query_id="q1", query="python", relevant_doc_ids=["d1"])]
    retriever = FakeRetriever({"python": ["d1", "d1", "d1", "d2"]})

    report = evaluate_retrieval(retriever, cases, k=4)

    assert report.cases[0].retrieved_doc_ids == ["d1", "d2"]
    assert report.metrics.precision_at_k == 0.25  # 1 relevant / k=4, not 3/4


def test_evaluateRetrieval_mapsStoredIdsBackToLabelIds() -> None:
    """Ingestion mints its own ids; scoring happens in the dataset's terms."""
    cases = [RetrievalCase(query_id="q1", query="python", relevant_doc_ids=["d1"])]
    retriever = FakeRetriever({"python": ["generated-uuid-1"]})

    report = evaluate_retrieval(retriever, cases, k=5, labels={"generated-uuid-1": "d1"})

    assert report.metrics.recall_at_k == 1.0


def test_evaluateRetrieval_validatesCorpusBeforeRunning() -> None:
    cases = [RetrievalCase(query_id="q1", query="q", relevant_doc_ids=["missing"])]
    with pytest.raises(ValueError, match="unknown doc_id"):
        evaluate_retrieval(FakeRetriever({}), cases, corpus=[CorpusDocument(doc_id="d1", text="x")])


def test_ingestCorpus_returnsStoredIdToLabelIdMapping() -> None:
    corpus = [CorpusDocument(doc_id="d1", text="alpha"), CorpusDocument(doc_id="d2", text="beta")]
    ingested: list[tuple[str, str]] = []

    def ingest(doc_id: str, text: str) -> str:
        ingested.append((doc_id, text))
        return f"stored-{doc_id}"

    mapping = ingest_corpus(corpus, ingest)

    assert ingested == [("d1", "alpha"), ("d2", "beta")]
    assert mapping == {"stored-d1": "d1", "stored-d2": "d2"}


# --- trust evaluation ------------------------------------------------------


def _trust_case(claim: str, expected: ClaimStatus) -> TrustCase:
    return TrustCase(
        case_id=claim[:8], claim=claim, evidence=["some evidence"], expected_status=expected
    )


def test_evaluateTrust_scoresVerdictsAgainstLabels() -> None:
    cases = [
        _trust_case("true claim", ClaimStatus.SUPPORTED),
        _trust_case("false claim", ClaimStatus.UNSUPPORTED),
    ]
    agent = FakeTrustAgent(
        {"true claim": [ClaimStatus.SUPPORTED], "false claim": [ClaimStatus.UNSUPPORTED]}
    )

    report = asyncio.run(evaluate_trust(agent, cases))

    assert report.metrics.accuracy == 1.0
    assert report.dangerous_errors == []


def test_evaluateTrust_tooLenientVerdict_flaggedAsDangerous() -> None:
    """Passing a fabrication is the error that reaches the user."""
    cases = [_trust_case("false claim", ClaimStatus.UNSUPPORTED)]
    agent = FakeTrustAgent({"false claim": [ClaimStatus.SUPPORTED]})

    report = asyncio.run(evaluate_trust(agent, cases))

    assert [c.case_id for c in report.dangerous_errors] == ["false cl"]


def test_evaluateTrust_tooStrictVerdict_isWrongButNotDangerous() -> None:
    """Flagging a true claim only costs a rewrite iteration."""
    cases = [_trust_case("true claim", ClaimStatus.SUPPORTED)]
    agent = FakeTrustAgent({"true claim": [ClaimStatus.UNSUPPORTED]})

    report = asyncio.run(evaluate_trust(agent, cases))

    assert report.metrics.accuracy == 0.0
    assert report.dangerous_errors == []


def test_evaluateTrust_splitClaim_takesTheWorstVerdict() -> None:
    """One bad part invalidates the claim as stated — it can't hide among good ones."""
    cases = [_trust_case("compound claim", ClaimStatus.PARTIALLY_SUPPORTED)]
    agent = FakeTrustAgent(
        {
            "compound claim": [
                ClaimStatus.SUPPORTED,
                ClaimStatus.PARTIALLY_SUPPORTED,
                ClaimStatus.SUPPORTED,
            ]
        }
    )

    report = asyncio.run(evaluate_trust(agent, cases))

    assert report.cases[0].predicted == ClaimStatus.PARTIALLY_SUPPORTED


def test_evaluateTrust_harnessExtractedNothing_countsAsUnsupportedNotSkipped() -> None:
    """'I found nothing to verify' is a harness failure, not a missing datapoint."""
    cases = [_trust_case("some claim", ClaimStatus.SUPPORTED)]
    agent = FakeTrustAgent({"some claim": []})

    report = asyncio.run(evaluate_trust(agent, cases))

    assert report.metrics.cases == 1
    assert report.cases[0].predicted == ClaimStatus.UNSUPPORTED


def test_ingestCorpus_duplicateContent_raisesRatherThanLosingALabel() -> None:
    """Ingestion dedups on cleaned text, so identical docs collapse to one id.

    Overwriting silently would make one label unreachable and understate
    recall forever — the same silent metric corruption validate_dataset
    guards against, reached through content instead of ids.
    """
    corpus = [
        CorpusDocument(doc_id="d1", text="same text"),
        CorpusDocument(doc_id="d2", text="same text"),
    ]

    with pytest.raises(ValueError, match="identical content"):
        ingest_corpus(corpus, lambda doc_id, text: "one-stored-id")
