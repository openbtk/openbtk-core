"""Unit tests for openbtk.guardrails.pipeline.GuardrailPipeline."""

from __future__ import annotations

from typing import Any

from openbtk.core.base import BaseGuardrail
from openbtk.core.schemas import GuardrailResult, GuardrailSeverity
from openbtk.guardrails.pipeline import GuardrailPipeline


class _FixedGuardrail(BaseGuardrail):
    """A real, minimal BaseGuardrail returning a fixed result -- used to
    drive GuardrailPipeline's own aggregation/short-circuit logic without
    depending on any real guardrail's specific detection behaviour."""

    def __init__(self, key: str, *, passed: bool, severity: GuardrailSeverity) -> None:
        self.registry_key = key
        self._passed = passed
        self._severity = severity
        self.call_count = 0

    def check(self, payload: Any) -> GuardrailResult:
        self.call_count += 1
        return GuardrailResult(
            passed=self._passed,
            severity=self._severity,
            guardrail_key=self.registry_key,
            message="ok" if self._passed else "failed",
        )


def _passing(key: str = "guardrail.general.a") -> _FixedGuardrail:
    return _FixedGuardrail(key, passed=True, severity=GuardrailSeverity.INFO)


def _blocking(key: str = "guardrail.general.b") -> _FixedGuardrail:
    return _FixedGuardrail(key, passed=False, severity=GuardrailSeverity.BLOCK)


def _warning(key: str = "guardrail.general.c") -> _FixedGuardrail:
    return _FixedGuardrail(key, passed=False, severity=GuardrailSeverity.WARNING)


class TestAllPass:
    def test_overall_passed_is_true(self) -> None:
        pipeline = GuardrailPipeline([_passing(), _passing("guardrail.general.d")])
        result = pipeline.run("payload")
        assert result.passed is True
        assert len(result.results) == 2
        assert result.short_circuited is False


class TestShortCircuit:
    def test_stops_after_the_first_block_by_default(self) -> None:
        first = _blocking()
        second = _passing()
        pipeline = GuardrailPipeline([first, second])
        result = pipeline.run("payload")
        assert result.passed is False
        assert result.short_circuited is True
        assert len(result.results) == 1
        assert second.call_count == 0

    def test_disabling_short_circuit_runs_every_guardrail(self) -> None:
        first = _blocking()
        second = _passing()
        pipeline = GuardrailPipeline([first, second], short_circuit=False)
        result = pipeline.run("payload")
        assert result.passed is False
        assert result.short_circuited is False
        assert len(result.results) == 2
        assert second.call_count == 1

    def test_a_warning_does_not_short_circuit(self) -> None:
        first = _warning()
        second = _passing()
        pipeline = GuardrailPipeline([first, second])
        result = pipeline.run("payload")
        assert len(result.results) == 2
        assert result.short_circuited is False
        # overall passed is still True -- only a failed BLOCK flips it.
        assert result.passed is True


class TestRegistryStringResolution:
    def test_a_string_entry_is_constructed_from_the_registry(self) -> None:
        from openbtk.guardrails.phi_leakage import PHILeakageGuardrail  # noqa: F401

        pipeline = GuardrailPipeline(["guardrail.general.phi_leakage"])
        result = pipeline.run("clean text")
        assert result.results[0].guardrail_key == "guardrail.general.phi_leakage"


class TestToGuardrailOutcomes:
    def test_reshapes_results_into_outcomes(self) -> None:
        pipeline = GuardrailPipeline(
            [_blocking(), _warning(), _passing()], short_circuit=False
        )
        result = pipeline.run("payload")
        outcomes = result.to_guardrail_outcomes(at="after:generate")
        assert len(outcomes) == 3
        assert all(o.at == "after:generate" for o in outcomes)
        assert all(o.checked_count == 1 for o in outcomes)
        assert outcomes[0].blocked_count == 1
        assert outcomes[0].warned_count == 0
        assert outcomes[1].blocked_count == 0
        assert outcomes[1].warned_count == 1
        assert outcomes[2].blocked_count == 0
        assert outcomes[2].warned_count == 0

    def test_sample_messages_are_empty_for_a_passed_result(self) -> None:
        pipeline = GuardrailPipeline([_passing()])
        result = pipeline.run("payload")
        outcomes = result.to_guardrail_outcomes(at="after:x")
        assert outcomes[0].sample_messages == []

    def test_sample_messages_carry_the_failure_message(self) -> None:
        pipeline = GuardrailPipeline([_blocking()])
        result = pipeline.run("payload")
        outcomes = result.to_guardrail_outcomes(at="after:x")
        assert outcomes[0].sample_messages == ["failed"]
