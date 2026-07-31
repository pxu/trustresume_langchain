"""Unit tests for the JSON logging setup."""

from __future__ import annotations

import json
import logging

from trustresume.logging_config import JsonFormatter, configure_logging


def _make_record(**extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="trustresume.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_jsonFormatter_rendersStandardFieldsAndMessage() -> None:
    payload = json.loads(JsonFormatter().format(_make_record()))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "trustresume.test"
    assert payload["message"] == "hello world"
    assert "timestamp" in payload


def test_jsonFormatter_includesExtraFields() -> None:
    payload = json.loads(JsonFormatter().format(_make_record(user_id="u1", iteration=2)))
    assert payload["user_id"] == "u1"
    assert payload["iteration"] == 2


def test_jsonFormatter_rendersExceptionInfo() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = _make_record()
        record.exc_info = sys.exc_info()
    payload = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in payload["exc_info"]


def test_loggerInfo_withExtra_doesNotRaiseOnReservedLookingKeys() -> None:
    """Regression test: ``extra`` keys must not collide with ``LogRecord``'s
    own attributes (``filename``, ``module``, ``name``, ``lineno``, ...) —
    stdlib ``logging`` raises ``KeyError`` if they do. Constructing a
    ``LogRecord`` by hand (as the other tests in this file do) skips that
    check entirely, so this test goes through the real ``Logger.info`` call
    path, the same way application code does.
    """
    logger = logging.getLogger("trustresume.test.extra")
    logger.addHandler(logging.NullHandler())
    logger.info("document ingested", extra={"user_id": "u1", "doc_filename": "resume.txt"})


def test_configureLogging_installsSingleJsonHandler() -> None:
    configure_logging("DEBUG")
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, JsonFormatter)
    assert root.level == logging.DEBUG

    # Idempotent: calling again doesn't stack handlers.
    configure_logging("WARNING")
    assert len(root.handlers) == 1
    assert root.level == logging.WARNING
