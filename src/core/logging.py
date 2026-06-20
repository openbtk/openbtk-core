"""
openbtk structured logging configuration.

Call `configure_logging()` once at application startup. Library code only
calls `structlog.get_logger(__name__)` — never configures the logger itself.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(
    level: str = "INFO",
    fmt: str = "json",
    redact_keys: list[str] | None = None,
) -> None:
    """Configure structlog for openbtk.

    Args:
        level: Log level string — "DEBUG", "INFO", "WARNING", "ERROR".
        fmt: Output format — "json" for production, "console" for development.
        redact_keys: Additional field keys to redact from log output. PHI
            keys (raw_text, patient_name, mrn, dob) are always redacted.
    """
    _always_redacted = {
        "raw_text", "patient_name", "mrn", "dob", "ssn",
        "address", "phone", "email", "patient_id_raw",
    }
    redact_set = _always_redacted | set(redact_keys or [])

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        _RedactProcessor(redact_set),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if fmt == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )


class _RedactProcessor:
    """Structlog processor that redacts sensitive field values."""

    _REDACTED = "[REDACTED]"

    def __init__(self, keys: set[str]) -> None:
        self._keys = keys

    def __call__(
        self, logger: Any, method: str, event_dict: dict[str, Any]
    ) -> dict[str, Any]:
        for key in self._keys:
            if key in event_dict:
                event_dict[key] = self._REDACTED
        return event_dict


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a bound structlog logger for the given module name.

    Usage in every module:
        import structlog
        log = structlog.get_logger(__name__)
    """
    return structlog.get_logger(name)
