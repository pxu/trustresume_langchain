"""SQLite connection factory and DDL for the structured store.

The schema here is the executable form of the SQLite half of
``architecture/data-model/README.md`` — keep the two in sync. Every table
carries a ``user_id`` (except ``users`` itself, which *is* the user) so
isolation can be enforced by filtering, per ADR-0001.

Milestone M2 (storage + retrieval).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# One statement per table. Executed with ``executescript`` in ``init_db``.
# ``IF NOT EXISTS`` makes initialization idempotent — safe to call on every
# startup.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    filename      TEXT NOT NULL,
    document_type TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_documents_user ON documents (user_id);
-- Enforces "same user, same content -> one document row" at the DB level,
-- not just in IngestionService's application-level check (belt and
-- suspenders — a concurrent duplicate ingest can't slip past a race in the
-- check-then-insert). content_hash is a hash of the *cleaned* text (see
-- ingestion/service.py), so trivial whitespace/encoding differences between
-- two uploads of "the same" document still count as a duplicate.
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_user_content
    ON documents (user_id, content_hash);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    document_id     TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    document_type   TEXT NOT NULL,
    source_document TEXT,
    text            TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chunks_user ON chunks (user_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks (document_id);

-- Full-text index over chunk text, for the keyword half of hybrid retrieval
-- (HybridRetriever). content='chunks'/content_rowid='rowid' makes this an
-- external-content FTS5 table: it indexes `chunks.text` without duplicating
-- it, keyed by `chunks`'s own SQLite rowid — chunk_id (the app-level id) is
-- looked up via that rowid at query time (see ChunkRepository.search_keywords).
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='rowid'
);

-- Triggers keep chunks_fts in sync with chunks automatically, so
-- ChunkRepository.add/delete_for_document don't need to know the FTS index
-- exists — the standard SQLite FTS5 external-content-table pattern.
CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts (rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts (chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts (chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO chunks_fts (rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TABLE IF NOT EXISTS generated_resumes (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    job_title    TEXT,
    iteration    INTEGER NOT NULL,
    summary      TEXT NOT NULL,
    content_json TEXT NOT NULL,
    trust_score  REAL NOT NULL,
    ats_score    REAL NOT NULL,
    passed       INTEGER NOT NULL,
    created_at   TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_resumes_user ON generated_resumes (user_id);

CREATE TABLE IF NOT EXISTS candidate_profiles (
    user_id      TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    doc_hash     TEXT NOT NULL,
    stale        INTEGER NOT NULL DEFAULT 0,
    updated_at   TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evaluations (
    id                TEXT PRIMARY KEY,
    resume_id         TEXT NOT NULL,
    user_id           TEXT NOT NULL,
    iteration         INTEGER NOT NULL,
    trust_score       REAL NOT NULL,
    ats_score         REAL NOT NULL,
    trust_report_json TEXT NOT NULL,
    ats_report_json   TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    FOREIGN KEY (resume_id) REFERENCES generated_resumes (id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_evaluations_resume ON evaluations (resume_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_user ON evaluations (user_id);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with the project's standard settings.

    - ``row_factory`` is ``sqlite3.Row`` so rows are addressable by column name.
    - Foreign keys are enabled per-connection (SQLite defaults them off), which
      makes the ``ON DELETE CASCADE`` relationships actually fire.
    - ``check_same_thread=False`` so the one shared connection can serve the
      FastAPI backend's threadpool workers. SQLite's default threading mode is
      serialized, and every repository write commits immediately, so this is
      safe for the single-connection, low-concurrency MVP; revisit (connection
      pool / per-request connection) if write concurrency grows.

    Pass ``":memory:"`` for an ephemeral database (used by the test suite).
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes if they don't already exist."""
    conn.executescript(_SCHEMA)
    conn.commit()
