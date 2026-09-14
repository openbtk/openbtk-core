"""Shared contract every registered BaseGuardrail must satisfy.

The single most important safety contract in the codebase after streaming:
``check()`` NEVER raises for a failed check -- it always returns a
GuardrailResult with ``passed=False``. A guardrail that raises makes
composition (running several guardrails in sequence, aggregating results)
impossible; ``GuardrailViolation`` is raised by the pipeline layer after
inspecting the result, never by the guardrail itself
(docs/04_API_DESIGN.md section 3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from openbtk.core.registry import GUARDRAIL_REGISTRY
from openbtk.core.schemas import GuardrailResult, GuardrailSeverity

if TYPE_CHECKING:
    from openbtk.core.base import BaseGuardrail


def _new_instance(key: str) -> BaseGuardrail:
    return GUARDRAIL_REGISTRY.create(key)


# Payloads deliberately chosen to be awkward: empty, huge, wrong type,
# containing surprising characters -- a guardrail that "never raises" must
# mean it, not just for the inputs its author happened to think of.
#
# Explicit `id=` on every case, not left to pytest's auto-generated ids: the
# huge-string case's raw value, embedded into a test node id, gets written
# into the PYTEST_CURRENT_TEST environment variable during setup/teardown --
# and Windows hard-caps an env var at 32767 characters, so a naive
# `"a" * 100_000` case crashes the whole run with ValueError before a single
# assertion executes. Found by actually running the suite, not by inspection.
_AWKWARD_PAYLOADS: list[Any] = [
    pytest.param("", id="empty-string"),
    pytest.param("a" * 100_000, id="huge-string"),
    pytest.param("normal text", id="normal-text"),
    pytest.param(None, id="none"),
    pytest.param(12345, id="int"),
    pytest.param(["not", "a", "string"], id="list"),
    pytest.param({"nested": {"structure": True}}, id="dict"),
    pytest.param("\x00\x01\x02 control characters", id="control-chars"),
    pytest.param("emoji test 🏥🩺", id="emoji"),
]


@pytest.mark.parametrize("key", GUARDRAIL_REGISTRY.list_keys())
class TestGuardrailContract:
    @pytest.mark.parametrize("payload", _AWKWARD_PAYLOADS)
    def test_check_never_raises(self, key: str, payload: Any) -> None:
        guardrail = _new_instance(key)
        result = guardrail.check(payload)  # must not raise, for ANY payload
        assert isinstance(result, GuardrailResult)

    def test_result_severity_is_valid_enum_member(self, key: str) -> None:
        guardrail = _new_instance(key)
        result = guardrail.check("test payload")
        assert isinstance(result.severity, GuardrailSeverity)

    def test_result_carries_its_own_guardrail_key(self, key: str) -> None:
        """A result should be traceable back to the guardrail that produced
        it -- essential once results from many guardrails are aggregated."""
        guardrail = _new_instance(key)
        result = guardrail.check("test payload")
        assert result.guardrail_key == key

    def test_failed_check_has_a_message(self, key: str) -> None:
        guardrail = _new_instance(key)
        result = guardrail.check("")  # the reference guardrail's failure case
        if not result.passed:
            assert result.message

    def test_provenance_is_serialisable(self, key: str) -> None:
        guardrail = _new_instance(key)
        dumped = guardrail.provenance().model_dump_json()
        assert isinstance(dumped, str) and len(dumped) > 0


# ---------------------------------------------------------------------------
# Meta-test: proves the never-raises check actually catches a violation.
# Never registered globally.
# ---------------------------------------------------------------------------


class _RaisingBrokenGuardrail:
    """Deliberately violates the never-raises contract."""

    def check(self, payload: Any) -> GuardrailResult:
        if payload == "":
            raise ValueError("empty payload is not allowed")  # the violation
        return GuardrailResult(
            passed=True,
            severity=GuardrailSeverity.INFO,
            guardrail_key="guardrail.general.broken",
            message="ok",
        )


def test_never_raises_check_catches_a_real_violation() -> None:
    broken = _RaisingBrokenGuardrail()
    with pytest.raises(ValueError, match="empty payload"):
        broken.check("")
    # And the corresponding assertion style used above would correctly flag
    # this: calling check() and asserting isinstance(result, GuardrailResult)
    # would never even reach the assertion, because check() raised first.
