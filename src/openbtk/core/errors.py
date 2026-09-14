"""OpenBTK exception hierarchy.

Every exception raised by OpenBTK's own code inherits from OpenBTKError and
carries structured, PHI-free context. This is the framework's single error
surface: modality modules, providers, and the pipeline layer all raise from
here rather than defining their own hierarchies.

Design contract (docs/04_API_DESIGN.md section 8):
    - Always chain wrapped exceptions with ``from e``, never swallow silently.
    - ``.context`` holds identifiers only -- a path, a URI, a hashed record id,
      a registry key. It never holds document content, patient data, or any
      value that could itself be PHI. This is enforced by
      ``tests/security/test_error_context.py``, not just documented here.
    - Error messages state what failed, why, and what to do next.
"""

from __future__ import annotations

from typing import Any


class OpenBTKError(Exception):
    """Root of every exception raised by OpenBTK.

    Args:
        message: Human-readable description of what failed, why, and what to
            do about it. Never includes PHI or raw document content.
        context: Structured, PHI-free metadata about where the failure
            occurred -- e.g. ``{"modality": "clinical_text", "stage": "load",
            "source": "/data/notes.csv"}``. Identifiers only, never content.

    Attributes:
        context: The context dict passed at construction, always a ``dict``
            (never ``None``) so callers can update it uniformly.

    Example:
        >>> try:
        ...     raise OpenBTKError("boom", context={"stage": "load"})
        ... except OpenBTKError as e:
        ...     e.context["stage"]
        'load'
    """

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = dict(context) if context is not None else {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self)!r}, context={self.context!r})"


class ConfigError(OpenBTKError):
    """A pipeline or component configuration is malformed or invalid."""


class RegistryError(OpenBTKError):
    """A registry operation failed: unknown key, duplicate key, or a class
    that does not subclass the registry's declared base type."""


class MissingDependencyError(OpenBTKError):
    """An optional dependency is required but not installed.

    Always names the extra to install (``pip install 'openbtk[<extra>]'``)
    rather than letting a bare ``ImportError`` propagate. Raised by
    ``openbtk.core._lazy.require()``.
    """


class PolicyError(OpenBTKError):
    """An operation is disallowed by the active pipeline policy.

    The primary case: constructing or invoking a provider whose
    ``sends_data_offsite`` is ``True`` while
    ``policy.allow_offsite_providers`` is ``False``.
    """


class LoaderError(OpenBTKError):
    """A loader failed to read or parse its source."""


class ProcessingError(OpenBTKError):
    """A preprocessor, chunker, segmenter, or feature extractor failed."""


class ProviderError(OpenBTKError):
    """An LLM or embedding provider failed to fulfil a request."""


class RateLimitError(ProviderError):
    """A provider rejected a request due to rate limiting.

    Distinguished from a generic ``ProviderError`` so callers can apply
    backoff-and-retry logic specifically to this case.
    """


class AuthenticationError(ProviderError):
    """A provider rejected a request due to invalid or missing credentials.

    Never includes the credential value itself in the message or context --
    only that authentication failed and which provider rejected it.
    """


class RetrievalError(OpenBTKError):
    """A vector store operation (upsert, query, delete) failed."""


class TerminologyError(OpenBTKError):
    """A terminology lookup failed: unavailable backend, unlicensed
    vocabulary, or an unresolvable code."""


class DatasetError(OpenBTKError):
    """A dataset adapter failed: missing credentials, bad path, or a
    network error while accessing a named dataset."""


class GuardrailViolation(OpenBTKError):  # noqa: N818 -- name fixed by docs/04_API_DESIGN.md §8
    """A guardrail result was severity BLOCK and the pipeline is configured
    to raise rather than continue.

    This is not a bug in the guardrail -- ``BaseGuardrail.check()`` never
    raises. This exception is raised by the pipeline layer after inspecting
    a ``GuardrailResult``, so it is meant to be caught at that level, not
    inside component code.
    """


class PluginError(OpenBTKError):
    """Entry-point plugin discovery or registration failed.

    Raised internally during plugin loading; ``load_plugins()`` catches
    these per-plugin and logs a warning rather than letting one broken
    third-party plugin prevent ``import openbtk`` from succeeding.
    """
