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
