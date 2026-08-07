"""Offline evaluation harness: is the system getting better, or just different?

Distinct from ``trustresume.evaluation`` (which scores one résumé for the user
at runtime): this package scores the *system* against labeled ground truth,
offline, so a change to chunking, retrieval, a prompt, or a model produces a
number that can be compared before and after.

Two suites, matching the two things that can silently regress:

* **Retrieval** (``retrieval_eval``) — recall@k / precision@k / MRR against a
  labeled corpus. Retrieval caps the whole pipeline: the writer can only
  ground claims in evidence that was actually surfaced.
* **Trust Harness** (``trust_eval``) — classification accuracy and macro-F1
  against labeled claims, plus a count of *too-lenient* verdicts, which are
  the errors that let a fabrication reach the user.

Run both: ``python -m trustresume.evals --suite all``. See ``evals/README.md``.
"""

from __future__ import annotations

from .datasets import CorpusDocument, RetrievalCase, TrustCase
from .metrics import ClassificationMetrics, RetrievalMetrics
from .retrieval_eval import RetrievalReport, evaluate_retrieval
from .trust_eval import TrustEvalReport, evaluate_trust

__all__ = [
    "ClassificationMetrics",
    "CorpusDocument",
    "RetrievalCase",
    "RetrievalMetrics",
    "RetrievalReport",
    "TrustCase",
    "TrustEvalReport",
    "evaluate_retrieval",
    "evaluate_trust",
]
