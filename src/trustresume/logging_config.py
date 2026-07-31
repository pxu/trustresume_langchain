"""Structured (JSON) logging setup, shared by the API and any CLI entry point.

Stdlib ``logging`` only — no new dependency for something this small. Emits one
JSON object per line (the conventional format for container log collectors:
CloudWatch, Datadog, `docker logs` piped to `jq`, etc.) so log fields stay
machine-parseable instead of living in an ad-hoc f-string.

Call :func:`configure_logging` once, at process start (``server.py``'s
``build_served_app``, or a script's ``main()``); library code below it just
calls ``logging.getLogger(__name__)`` as usual and never configures handlers
itself, so importing this package has no side effects.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message"}


class JsonFormatter(logging.Formatter):
    """Renders each ``LogRecord`` as one JSON object per line.

    Standard fields (``timestamp``/``level``/``logger``/``message``) plus
    whatever extra keyword arguments the caller passed via
    ``logger.info(..., extra={...})`` — e.g. ``user_id``, ``iteration``,
    ``node`` — so structured context survives into the log line instead of
    being flattened into free text.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED}
        payload.update(extras)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str | None = None) -> None:
    """Install a single JSON-formatted stream handler on the root logger.

    Idempotent: safe to call more than once (e.g. once from ``server.py`` and
    once from a test) — repeat calls just reset the handler list rather than
    stacking duplicate handlers. ``level`` defaults to ``$TRUSTRESUME_LOG_LEVEL``
    or ``INFO``.
    """
    resolved = (level or os.getenv("TRUSTRESUME_LOG_LEVEL") or "INFO").upper()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved)
