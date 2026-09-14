"""Unit tests for the openbtk.core.errors hierarchy."""

from __future__ import annotations

import pytest

from openbtk.core.errors import (
    AuthenticationError,
    ConfigError,
    DatasetError,
    GuardrailViolation,
    LoaderError,
    MissingDependencyError,
    OpenBTKError,
    PluginError,
    PolicyError,
    ProcessingError,
    ProviderError,
    RateLimitError,
    RegistryError,
    RetrievalError,
    TerminologyError,
)

_ALL_SUBCLASSES = [
    ConfigError,
    RegistryError,
    MissingDependencyError,
    PolicyError,
    LoaderError,
    ProcessingError,
    ProviderError,
    RateLimitError,
    AuthenticationError,
    RetrievalError,
    TerminologyError,
    DatasetError,
    GuardrailViolation,
    PluginError,
]


@pytest.mark.parametrize("exc_cls", _ALL_SUBCLASSES)
def test_every_error_is_an_openbtk_error(exc_cls: type[OpenBTKError]) -> None:
    assert issubclass(exc_cls, OpenBTKError)


def test_context_defaults_to_an_empty_dict_not_none() -> None:
    """Callers must be able to update .context uniformly, without a None check."""
    err = OpenBTKError("message")
    assert err.context == {}


def test_context_is_a_copy_not_the_caller_s_original_dict() -> None:
    """Mutating the caller's dict after construction must not affect the
    stored context -- OpenBTKError.__init__ does `dict(context)`."""
    original = {"key": "value"}
    err = OpenBTKError("message", context=original)
    original["key"] = "mutated"
    assert err.context["key"] == "value"


def test_message_is_preserved_via_str() -> None:
    err = OpenBTKError("something specific failed")
    assert str(err) == "something specific failed"


def test_repr_includes_class_name_and_context() -> None:
    err = ConfigError("bad config", context={"path": "/tmp/x.yaml"})
    r = repr(err)
    assert "ConfigError" in r
    assert "path" in r


def test_chaining_preserves_the_original_exception() -> None:
    """The coding standard (docs/09_CODING_STANDARDS.md section 8) requires
    `raise X from e` everywhere -- verify the mechanism itself works as
    intended for our own hierarchy."""
    original = ValueError("root cause")
    try:
        try:
            raise original
        except ValueError as e:
            raise LoaderError("wrapped", context={"source": "test"}) from e
    except LoaderError as wrapped:
        assert wrapped.__cause__ is original


def test_rate_limit_and_authentication_are_provider_errors() -> None:
    """These two exist specifically so callers can catch them separately
    from a generic ProviderError (retry-with-backoff vs. fail-fast)."""
    assert issubclass(RateLimitError, ProviderError)
    assert issubclass(AuthenticationError, ProviderError)
    assert not issubclass(RateLimitError, AuthenticationError)
    assert not issubclass(AuthenticationError, RateLimitError)


def test_rate_limit_error_context_never_needs_to_carry_credentials() -> None:
    """A regression guard, not a design constraint on the class itself:
    nothing about RateLimitError/AuthenticationError requires or encourages
    passing a credential value into .context. Constructing one with only
    identifiers must work without any special-casing."""
    err = AuthenticationError(
        "authentication failed", context={"provider": "openai_compatible"}
    )
    assert err.context == {"provider": "openai_compatible"}
