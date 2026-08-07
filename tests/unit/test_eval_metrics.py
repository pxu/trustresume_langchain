"""Unit tests for the offline evaluation metrics.

Pure functions over hand-computed expectations — the point of a metric is that
it's *checkable*, so every assertion here uses a number derivable by hand from
the inputs rather than a golden value copied out of a previous run.
"""

from __future__ import annotations

import pytest

from trustresume.evals.metrics import (
    aggregate_retrieval,
    classification_metrics,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# --- retrieval -------------------------------------------------------------


def test_recallAtK_partialHits_isFractionOfRelevantFound() -> None:
    assert recall_at_k(["a", "b", "c"], ["a", "d"], k=3) == 0.5


def test_recallAtK_ignoresHitsBelowTheCutOff() -> None:
    """A document ranked 6th doesn't reach the prompt when only 5 are used."""
    assert recall_at_k(["x", "x", "x", "x", "x", "a"], ["a"], k=5) == 0.0


def test_recallAtK_noRelevantDocs_isPerfectNotDivideByZero() -> None:
    """An unanswerable query can't be 'missed' — nothing was there to find."""
    assert recall_at_k(["a"], [], k=5) == 1.0


def test_precisionAtK_dividesByKNotByResultCount() -> None:
    """Returning 1 good result out of a possible 5 wastes 4 context slots."""
    assert precision_at_k(["a"], ["a"], k=5) == 0.2


def test_precisionAtK_zeroK_isZeroNotDivideByZero() -> None:
    assert precision_at_k(["a"], ["a"], k=0) == 0.0


def test_reciprocalRank_usesFirstRelevantPosition() -> None:
    assert reciprocal_rank(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)


def test_reciprocalRank_noHit_isZero() -> None:
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_aggregateRetrieval_macroAveragesAcrossQueries() -> None:
    """Each query counts once, regardless of how many labels it carries."""
    results = [
        (["a", "b"], ["a"]),  # recall 1.0, RR 1.0
        (["x", "y"], ["z"]),  # recall 0.0, RR 0.0
    ]
    metrics = aggregate_retrieval(results, k=2)

    assert metrics.queries == 2
    assert metrics.recall_at_k == 0.5
    assert metrics.mrr == 0.5
    assert metrics.hit_rate == 0.5


def test_aggregateRetrieval_unanswerableQuery_excludedFromPrecisionAndHitRate() -> None:
    """A query with nothing to find can't be 'missed' — scoring it as a miss
    would penalize correct behavior and make the headline numbers depend on
    how many such cases the dataset contains."""
    results = [
        (["a"], ["a"]),  # answerable, perfect
        (["x", "y"], []),  # unanswerable, returned 2 results anyway
    ]
    metrics = aggregate_retrieval(results, k=1)

    assert metrics.queries == 2
    assert metrics.hit_rate == 1.0  # scored over the one answerable query only
    assert metrics.precision_at_k == 1.0
    assert metrics.unanswerable_queries == 1
    assert metrics.unanswerable_results == 1.0  # k=1 truncates the 2 returned


def test_aggregateRetrieval_noQueries_returnsZeroedMetrics() -> None:
    metrics = aggregate_retrieval([], k=5)
    assert metrics.queries == 0
    assert metrics.recall_at_k == 0.0


# --- classification --------------------------------------------------------


def test_classificationMetrics_accuracyAndPerLabelScores() -> None:
    expected = ["SUPPORTED", "SUPPORTED", "UNSUPPORTED", "UNSUPPORTED"]
    predicted = ["SUPPORTED", "UNSUPPORTED", "UNSUPPORTED", "UNSUPPORTED"]

    metrics = classification_metrics(expected, predicted)

    assert metrics.cases == 4
    assert metrics.accuracy == 0.75
    by_label = {m.label: m for m in metrics.per_label}
    # SUPPORTED: 1 of 1 predicted correct (P=1.0), 1 of 2 actual found (R=0.5).
    assert by_label["SUPPORTED"].precision == 1.0
    assert by_label["SUPPORTED"].recall == 0.5
    # UNSUPPORTED: 2 of 3 predicted correct, 2 of 2 actual found.
    assert by_label["UNSUPPORTED"].precision == pytest.approx(2 / 3)
    assert by_label["UNSUPPORTED"].recall == 1.0


def test_classificationMetrics_alwaysPredictingMajority_exposedByMacroF1() -> None:
    """The failure macro-F1 exists to catch: high accuracy, useless model.

    A harness that calls everything SUPPORTED scores 80% accuracy on this
    distribution while never once catching a fabrication.
    """
    expected = ["SUPPORTED"] * 8 + ["UNSUPPORTED"] * 2
    predicted = ["SUPPORTED"] * 10

    metrics = classification_metrics(expected, predicted)

    assert metrics.accuracy == 0.8
    assert metrics.macro_f1 < 0.5  # macro-F1 refuses to be fooled


def test_classificationMetrics_confusionShowsErrorDirection() -> None:
    metrics = classification_metrics(["UNSUPPORTED"], ["SUPPORTED"])
    assert metrics.confusion["UNSUPPORTED"]["SUPPORTED"] == 1


def test_classificationMetrics_mismatchedLengths_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        classification_metrics(["A"], ["A", "B"])


def test_classificationMetrics_noCases_returnsZeroed() -> None:
    metrics = classification_metrics([], [])
    assert metrics.cases == 0
    assert metrics.accuracy == 0.0
