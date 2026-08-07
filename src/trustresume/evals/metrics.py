"""Offline evaluation metrics: retrieval quality and classification quality.

Pure functions over labeled data — no LLM, no I/O, no framework. They answer
the question the runtime quality gate cannot: *is the system getting better?*

The distinction matters and is easy to blur. ``trustresume.evaluation`` scores
one generated résumé for the user (product logic, shipped in the response).
This module scores the *system* against ground truth (engineering logic, run
offline against a labeled dataset). A change to chunking, the embedding model,
the fusion weights, or a prompt moves these numbers; nothing here is ever
shown to an end user.

Added post-port; no equivalent in the original.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field


class RetrievalMetrics(BaseModel):
    """Ranking quality for one query set, at a fixed cut-off ``k``."""

    model_config = ConfigDict(extra="forbid")

    k: int = Field(..., ge=1, description="Cut-off the metrics were computed at.")
    queries: int = Field(..., ge=0, description="Number of queries scored.")
    recall_at_k: float = Field(..., ge=0, le=1, description="Mean fraction of relevant docs found.")
    precision_at_k: float = Field(..., ge=0, le=1, description="Mean fraction of hits that matter.")
    mrr: float = Field(..., ge=0, le=1, description="Mean reciprocal rank of the first hit.")
    hit_rate: float = Field(..., ge=0, le=1, description="Fraction of queries with >=1 hit in k.")
    unanswerable_queries: int = Field(
        0, ge=0, description="Queries with no relevant document (excluded from P/MRR/hit rate)."
    )
    unanswerable_results: float = Field(
        0.0,
        ge=0,
        description="Mean results returned for unanswerable queries — lower is better.",
    )


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of the relevant documents that appear in the top ``k``.

    The headline RAG metric: a generator can only ground a claim in evidence
    that retrieval actually surfaced, so recall caps the whole pipeline's
    ceiling. A query with no relevant documents scores 1.0 — nothing was
    missed — rather than dividing by zero.
    """
    if not relevant:
        return 1.0
    found = len(set(retrieved[:k]) & set(relevant))
    return found / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of the top ``k`` results that are relevant.

    Divided by ``k``, not by the number retrieved: returning 2 results when
    ``k=5`` should not score the same as returning 5 good ones, because the
    prompt has room for 5 and the three empty slots are wasted context.
    """
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / k


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """``1/rank`` of the first relevant hit, or 0.0 if there is none.

    Complements recall: recall says "was the evidence there at all", RR says
    "how far down". Rank matters because an LLM attends unevenly across a long
    context, so evidence buried at position 9 is not as good as position 1
    even when both are "retrieved".
    """
    relevant_set = set(relevant)
    for index, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / index
    return 0.0


def aggregate_retrieval(
    results: Sequence[tuple[Sequence[str], Sequence[str]]], k: int
) -> RetrievalMetrics:
    """Mean metrics over ``(retrieved, relevant)`` pairs — one pair per query.

    Macro-averaged (each query weighted equally) rather than micro-averaged
    (each relevant document weighted equally): a single query with ten labeled
    documents shouldn't dominate the score of a suite where most queries have
    one or two.

    **Queries with no relevant document are excluded from precision, MRR, and
    hit rate.** Those metrics are undefined when there is nothing to find —
    counting such a query as a miss would penalize the system for correctly
    having no answer, and would make the suite's headline numbers depend on
    how many unanswerable cases the dataset happens to contain (this is
    standard IR practice; TREC does the same). They still count for recall,
    where "missed nothing" is genuinely 1.0, and their real signal — *did the
    system return junk anyway* — is reported separately as
    ``unanswerable_results``.
    """
    if not results:
        return RetrievalMetrics(
            k=k, queries=0, recall_at_k=0.0, precision_at_k=0.0, mrr=0.0, hit_rate=0.0
        )
    answerable = [(r, rel) for r, rel in results if rel]
    unanswerable = [r for r, rel in results if not rel]

    recalls = [recall_at_k(r, rel, k) for r, rel in results]
    precisions = [precision_at_k(r, rel, k) for r, rel in answerable]
    rrs = [reciprocal_rank(r[:k], rel) for r, rel in answerable]
    hits = [1.0 if set(r[:k]) & set(rel) else 0.0 for r, rel in answerable]

    scored = len(answerable)
    return RetrievalMetrics(
        k=k,
        queries=len(results),
        recall_at_k=sum(recalls) / len(results),
        precision_at_k=sum(precisions) / scored if scored else 0.0,
        mrr=sum(rrs) / scored if scored else 0.0,
        hit_rate=sum(hits) / scored if scored else 0.0,
        unanswerable_queries=len(unanswerable),
        unanswerable_results=(
            sum(len(r[:k]) for r in unanswerable) / len(unanswerable) if unanswerable else 0.0
        ),
    )


class LabelMetrics(BaseModel):
    """Precision/recall/F1 for a single class."""

    model_config = ConfigDict(extra="forbid")

    label: str
    support: int = Field(..., ge=0, description="How many cases truly have this label.")
    predicted: int = Field(..., ge=0, description="How many cases were predicted this label.")
    precision: float = Field(..., ge=0, le=1)
    recall: float = Field(..., ge=0, le=1)
    f1: float = Field(..., ge=0, le=1)


class ClassificationMetrics(BaseModel):
    """Accuracy plus per-label and macro-averaged precision/recall/F1."""

    model_config = ConfigDict(extra="forbid")

    cases: int = Field(..., ge=0)
    accuracy: float = Field(..., ge=0, le=1)
    macro_f1: float = Field(..., ge=0, le=1)
    per_label: list[LabelMetrics] = Field(default_factory=list)
    confusion: dict[str, dict[str, int]] = Field(
        default_factory=dict, description="confusion[expected][predicted] = count."
    )


def _f1(precision: float, recall: float) -> float:
    """Harmonic mean, 0.0 when both are 0 (rather than dividing by zero)."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def classification_metrics(
    expected: Sequence[str], predicted: Sequence[str]
) -> ClassificationMetrics:
    """Score a multi-class classifier against ground truth.

    Reports macro-F1 alongside accuracy because the label distribution here is
    inherently skewed — most claims in a real draft are SUPPORTED, so a model
    that never predicts UNSUPPORTED can look ~80% accurate while failing at
    the single job the Trust Harness exists to do. Macro-F1 weights the rare
    classes equally and exposes that; the confusion matrix shows *which*
    direction the errors go, which is what tells you whether the harness is
    too lenient (the dangerous direction) or too strict.
    """
    if len(expected) != len(predicted):
        raise ValueError(
            f"expected and predicted must be the same length; "
            f"got {len(expected)} and {len(predicted)}"
        )
    if not expected:
        return ClassificationMetrics(cases=0, accuracy=0.0, macro_f1=0.0)

    labels = sorted(set(expected) | set(predicted))
    confusion: dict[str, dict[str, int]] = {
        label: dict.fromkeys(labels, 0) for label in sorted(set(expected))
    }
    for exp, pred in zip(expected, predicted, strict=True):
        confusion[exp][pred] += 1

    per_label: list[LabelMetrics] = []
    for label in labels:
        true_positive = sum(1 for e, p in zip(expected, predicted, strict=True) if e == p == label)
        support = sum(1 for e in expected if e == label)
        predicted_count = sum(1 for p in predicted if p == label)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        per_label.append(
            LabelMetrics(
                label=label,
                support=support,
                predicted=predicted_count,
                precision=precision,
                recall=recall,
                f1=_f1(precision, recall),
            )
        )

    correct = sum(1 for e, p in zip(expected, predicted, strict=True) if e == p)
    # Macro-average over labels that actually occur in the ground truth: a
    # label the model hallucinated but that no case truly has would otherwise
    # drag macro-F1 down twice (once as its own zero, once via the wrong
    # predictions it caused).
    scored = [m for m in per_label if m.support > 0]
    macro_f1 = sum(m.f1 for m in scored) / len(scored) if scored else 0.0
    return ClassificationMetrics(
        cases=len(expected),
        accuracy=correct / len(expected),
        macro_f1=macro_f1,
        per_label=per_label,
        confusion=confusion,
    )
