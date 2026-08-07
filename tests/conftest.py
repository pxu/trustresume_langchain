"""Shared pytest fixtures for TrustResume (unit + integration tests).

The two fixtures here let the storage/retrieval/ingestion tests run fully
in-process: an in-memory SQLite connection and a deterministic fake embedder
(no fastembed model download, no network, no flakiness). The fake's class lives
in ``tests/fakes.py`` so it can also be imported directly by test modules.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest

from tests.fakes import FakeEmbeddings
from trustresume.storage import connect, init_db


@pytest.fixture(autouse=True)
def _isolate_pricing_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep cost assertions independent of the developer's environment.

    ``track_usage()`` calls ``load_pricing()`` on every orchestrator run, and
    that reads ``$TRUSTRESUME_PRICING``. A machine with it pointed at a table
    containing the fake models\' ids would flip the suite\'s
    ``cost_usd is None`` assertions to a number — a failure that reproduces on
    one laptop and nowhere else.
    """
    monkeypatch.delenv("TRUSTRESUME_PRICING", raising=False)


@pytest.fixture
def db() -> Iterator[sqlite3.Connection]:
    """A fresh, initialized in-memory SQLite database per test."""
    conn = connect(":memory:")
    init_db(conn)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def fake_embedder() -> FakeEmbeddings:
    """A deterministic embedder implementing ``langchain_core.embeddings.Embeddings``."""
    return FakeEmbeddings()
