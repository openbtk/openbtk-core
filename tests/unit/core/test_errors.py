"""Unit tests for openbtk.core.errors."""
from __future__ import annotations

import pytest

from openbtk.core.errors import (
    openbtkError,
    ConfigError,
    LoaderError,
    ProviderError,
    ProcessingError,
    GuardrailViolation,
    RegistryError,
)


def test_base_error_without_context() -> None:
    err = openbtkError("something failed")
    assert str(err) == "something failed"
    assert err.context == {}


def test_base_error_with_context_in_str() -> None:
    err = openbtkError(
        "load failed", context={"modality": "imaging", "source": "scan.dcm"}
    )
    text = str(err)
    assert "load failed" in text
    assert "modality='imaging'" in text
    assert "source='scan.dcm'" in text


def test_subclasses_inherit_base() -> None:
    assert issubclass(ConfigError, openbtkError)
    assert issubclass(LoaderError, openbtkError)
    assert issubclass(ProviderError, openbtkError)
    assert issubclass(ProcessingError, openbtkError)
    assert issubclass(GuardrailViolation, openbtkError)
    assert issubclass(RegistryError, openbtkError)


def test_guardrail_violation_carries_details() -> None:
    err = GuardrailViolation(
        "PHI detected",
        details={"entity_types": ["PERSON", "DATE"]},
        context={"record_id": "abc123"},
    )
    assert err.details == {"entity_types": ["PERSON", "DATE"]}
    assert err.context == {"record_id": "abc123"}
    assert "PHI detected" in str(err)


def test_error_chaining_preserves_cause() -> None:
    try:
        try:
            raise ValueError("original failure")
        except ValueError as e:
            raise LoaderError("wrapped failure", context={"source": "x.csv"}) from e
    except LoaderError as wrapped:
        assert wrapped.__cause__ is not None
        assert isinstance(wrapped.__cause__, ValueError)
        assert str(wrapped.__cause__) == "original failure"


def test_raises_as_expected_exception_type() -> None:
    with pytest.raises(ProcessingError):
        raise ProcessingError("bad chunk")
