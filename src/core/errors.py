"""
openbtk error hierarchy.

All framework exceptions inherit from openbtkError, carrying structured
context for observability. Never raise bare built-ins from library code.
"""
from __future__ import annotations

from typing import Any


class openbtkError(Exception):
    """Base exception for all openbtk errors."""

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = context or {}

    def __str__(self) -> str:
        base = super().__str__()
        if self.context:
            ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{base} [{ctx}]"
        return base


class ConfigError(openbtkError):
    """Invalid or missing configuration."""


class LoaderError(openbtkError):
    """Failure during data loading from a source."""


class ProviderError(openbtkError):
    """Failure from an LLM, embedding, or external API provider."""


class ProcessingError(openbtkError):
    """Failure during preprocessing, chunking, or feature extraction."""


class GuardrailViolation(openbtkError):
    """A guardrail check determined the payload should be blocked."""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, context)
        self.details: dict[str, Any] = details or {}


class RegistryError(openbtkError):
    """Registry key not found or duplicate registration."""


class PluginError(openbtkError):
    """Failure during plugin discovery or loading."""


class DatasetError(openbtkError):
    """Failure during dataset loading or access (e.g., missing credentials)."""


class EmbeddingError(openbtkError):
    """Failure during embedding generation."""


class RetrievalError(openbtkError):
    """Failure during vector store upsert or query."""
