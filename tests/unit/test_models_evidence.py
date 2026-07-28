"""Unit tests for the Evidence schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trustresume.models import DocumentType, EvidenceChunk, EvidenceSet


def _chunk(**overrides: object) -> EvidenceChunk:
    base: dict[str, object] = {
        "chunk_id": "c1",
        "user_id": "u1",
        "document_id": "d1",
        "text": "Built a Python service on AWS.",
    }
    base.update(overrides)
    return EvidenceChunk(**base)  # type: ignore[arg-type]


def test_evidenceChunk_minimalInput_appliesDefaults() -> None:
    chunk = _chunk()

    assert chunk.document_type is DocumentType.OTHER
    assert chunk.source_document is None
    assert chunk.score is None


def test_evidenceChunk_requiresUserId() -> None:
    with pytest.raises(ValidationError):
        _chunk(user_id="")


def test_evidenceSet_defaultsToNoChunks() -> None:
    ev = EvidenceSet(user_id="u1", query="python aws")

    assert ev.chunks == []


def test_evidenceSet_carriesChunks() -> None:
    ev = EvidenceSet(
        user_id="u1",
        query="python",
        chunks=[_chunk(document_type=DocumentType.RESUME, score=0.91)],
    )

    assert ev.chunks[0].document_type is DocumentType.RESUME
    assert ev.chunks[0].score == pytest.approx(0.91)
