"""Structured JSON logging.

Emitting logs as single-line JSON makes them trivially parseable by log
aggregators (CloudWatch, Loki, ELK). Arbitrary context can be attached via the
standard ``extra=`` mechanism, e.g. ``logger.info("ingested", extra={...})``.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

# Attributes present on every LogRecord; anything else is treated as structured
# context supplied through ``extra=`` and merged into the JSON payload.
_STANDARD_ATTRS = set(
    vars(logging.makeLogRecord({})).keys()
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """Render a :class:`logging.LogRecord` as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger (idempotent)."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    """Return a module logger."""
    return logging.getLogger(name)
