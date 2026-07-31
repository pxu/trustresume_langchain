"""Unit tests for the API's wire schemas (``trustresume.api.schemas``).

``GenerateResponse``/``SearchResponse`` are mostly exercised indirectly via
the integration tests in ``tests/integration/test_api.py``; this file covers
the projection logic (``from_state``/``from_evidence``) directly, including
the error path a full HTTP round trip can't easily trigger.
"""

from __future__ import annotations

import pytest

from trustresume.api.schemas import GenerateResponse, SearchResponse
from trustresume.models import EvidenceChunk, EvidenceSet, WorkflowState


def test_generateResponse_fromState_noScoredDraft_raises() -> None:
    empty_state = WorkflowState(user_id="u1")  # no drafts/trust_reports/ats_reports
    with pytest.raises(ValueError, match="no scored draft"):
        GenerateResponse.from_state(empty_state)


def test_searchResponse_fromEvidence_projectsChunks() -> None:
    evidence = EvidenceSet(
        user_id="u1",
        query="python aws",
        chunks=[
            EvidenceChunk(
                chunk_id="c1",
                user_id="u1",
                document_id="d1",
                source_document="resume.txt",
                text="Built Python services on AWS.",
                score=0.87,
            )
        ],
    )

    response = SearchResponse.from_evidence(evidence)

    assert response.query == "python aws"
    assert len(response.chunks) == 1
    chunk = response.chunks[0]
    assert chunk.chunk_id == "c1"
    assert chunk.source_document == "resume.txt"
    assert chunk.score == 0.87


def test_searchResponse_fromEvidence_noChunks() -> None:
    evidence = EvidenceSet(user_id="u1", query="nothing matches")
    response = SearchResponse.from_evidence(evidence)
    assert response.chunks == []
