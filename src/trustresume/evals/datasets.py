"""Labeled evaluation data: typed records plus JSONL loading.

The datasets live outside the package, in ``evals/datasets/*.jsonl`` at the
repo root, for the same reason ``config/llm.json`` does: they're meant to be
read, reviewed, and extended by a human, and a diff of a JSONL file is
reviewable in a way a Python literal buried in ``src/`` isn't.

JSONL (not JSON) because these files only ever grow by appending cases, and
one-record-per-line keeps that a one-line diff instead of a re-indented block.

Added post-port; no equivalent in the original.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from trustresume.models import ClaimCategory, ClaimStatus

#: Repo-root ``evals/datasets/``. Resolved relative to this file the same way
#: ``model_factory.DEFAULT_CONFIG_PATH`` resolves ``config/``.
DATASETS_DIR = Path(__file__).resolve().parents[3] / "evals" / "datasets"


class CorpusDocument(BaseModel):
    """One document in the retrieval evaluation corpus."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class RetrievalCase(BaseModel):
    """A query plus the documents a correct system must retrieve for it."""

    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    relevant_doc_ids: list[str] = Field(default_factory=list)
    note: str | None = Field(None, description="Why this case exists — what capability it probes.")


class TrustCase(BaseModel):
    """A single claim, the evidence available, and the correct verdict.

    One claim per case (rather than a whole draft with many) so a wrong
    verdict is attributable to exactly one judgment, and so the label is
    something a human can assign unambiguously.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(..., min_length=1)
    claim: str = Field(..., min_length=1)
    evidence: list[str] = Field(default_factory=list)
    expected_status: ClaimStatus
    category: ClaimCategory = Field(
        default=ClaimCategory.OTHER, description="Claim type, as the harness would categorize it."
    )
    note: str | None = Field(None, description="Why this case exists — what failure it guards.")


def _read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    """Yield each non-blank line of a JSONL file as a dict.

    Blank lines and ``#`` comment lines are skipped so a dataset can carry
    section headers explaining what a block of cases probes — strict JSONL
    has no comment syntax, and an uncommentable dataset stops getting
    extended.
    """
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parsed: dict[str, object] = json.loads(stripped)
            yield parsed


def load_corpus(path: Path | None = None) -> list[CorpusDocument]:
    """The retrieval corpus (``retrieval_corpus.jsonl`` by default)."""
    source = path or DATASETS_DIR / "retrieval_corpus.jsonl"
    return [CorpusDocument.model_validate(row) for row in _read_jsonl(source)]


def load_retrieval_cases(path: Path | None = None) -> list[RetrievalCase]:
    """The retrieval queries + relevance labels (``retrieval_queries.jsonl``)."""
    source = path or DATASETS_DIR / "retrieval_queries.jsonl"
    return [RetrievalCase.model_validate(row) for row in _read_jsonl(source)]


def load_trust_cases(path: Path | None = None) -> list[TrustCase]:
    """The Trust Harness classification cases (``trust_claims.jsonl``)."""
    source = path or DATASETS_DIR / "trust_claims.jsonl"
    return [TrustCase.model_validate(row) for row in _read_jsonl(source)]


def validate_dataset(corpus: list[CorpusDocument], cases: list[RetrievalCase]) -> None:
    """Fail loudly on labels that reference documents the corpus doesn't have.

    A typo'd ``doc_id`` is otherwise invisible and *silently lowers recall*
    forever — the system is asked to retrieve something that cannot exist, so
    the metric under-reports and every later comparison inherits the error.
    Cheap to check, expensive to discover months later.
    """
    known = {doc.doc_id for doc in corpus}
    duplicates = len(corpus) - len(known)
    if duplicates:
        raise ValueError(f"retrieval corpus has {duplicates} duplicate doc_id(s)")
    for case in cases:
        unknown = set(case.relevant_doc_ids) - known
        if unknown:
            raise ValueError(f"case {case.query_id!r} labels unknown doc_id(s): {sorted(unknown)}")
