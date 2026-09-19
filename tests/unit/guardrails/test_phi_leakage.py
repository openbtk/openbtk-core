"""Unit tests for openbtk.guardrails.phi_leakage.PHILeakageGuardrail."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from openbtk.core.schemas import GuardrailSeverity
from openbtk.deid.schemas import DeidReport, DeidResult, PHICategory, RiskEstimate
from openbtk.guardrails.phi_leakage import PHILeakageGuardrail

# Defined once with the suppression marker, then referenced by name
# everywhere below -- tests/security/test_fixture_hygiene.py scans by
# line text, so a line using the constant's NAME never re-trips the
# scanner (the same established convention as
# tests/integration/test_clinical_text_pipeline.py).
_SSN_VALUE = "123-45-6789"  # phi-fixture-ok: synthetic, unassigned test value
_SSN_TEXT = f"Patient SSN: {_SSN_VALUE}."


class TestNoPHI:
    def test_passes_for_clean_text(self) -> None:
        guardrail = PHILeakageGuardrail()
        result = guardrail.check("No identifiers here.")
        assert result.passed is True
        assert result.severity == GuardrailSeverity.INFO

    def test_passes_for_empty_string(self) -> None:
        guardrail = PHILeakageGuardrail()
        assert guardrail.check("").passed is True


class TestPHIDetected:
    def test_ssn_is_detected_and_blocks(self) -> None:
        guardrail = PHILeakageGuardrail()
        result = guardrail.check(_SSN_TEXT)
        assert result.passed is False
        assert result.severity == GuardrailSeverity.BLOCK

    def test_message_never_contains_the_detected_value(self) -> None:
        guardrail = PHILeakageGuardrail()
        result = guardrail.check(_SSN_TEXT)
        assert _SSN_VALUE not in result.message
        assert _SSN_VALUE not in str(result.details)

    def test_spans_carry_offsets_not_content(self) -> None:
        guardrail = PHILeakageGuardrail()
        result = guardrail.check(_SSN_TEXT)
        assert len(result.spans) >= 1
        assert all(hasattr(s, "start") and hasattr(s, "end") for s in result.spans)

    def test_details_include_categories_and_count(self) -> None:
        guardrail = PHILeakageGuardrail()
        result = guardrail.check(_SSN_TEXT)
        assert result.details["count"] >= 1
        assert isinstance(result.details["categories"], list)


class TestPayloadShapes:
    def test_accepts_a_record_like_object_with_a_text_attribute(self) -> None:
        class _Record:
            text = _SSN_TEXT

        guardrail = PHILeakageGuardrail()
        result = guardrail.check(_Record())
        assert result.passed is False

    def test_non_string_non_record_payload_is_coerced_via_str(self) -> None:
        guardrail = PHILeakageGuardrail()
        result = guardrail.check(12345)
        assert result.passed is True

    def test_none_payload_never_raises(self) -> None:
        guardrail = PHILeakageGuardrail()
        result = guardrail.check(None)
        assert result.passed is True


class TestEngineFailureNeverRaises:
    def test_an_internal_engine_exception_becomes_a_warning_result(self) -> None:
        guardrail = PHILeakageGuardrail()
        broken_engine = MagicMock()
        broken_engine.deidentify.side_effect = RuntimeError("boom")
        guardrail._engine = broken_engine
        result = guardrail.check("anything")
        assert result.passed is False
        assert result.severity == GuardrailSeverity.WARNING


class TestRegistration:
    def test_registered_under_the_expected_key(self) -> None:
        assert PHILeakageGuardrail.registry_key == "guardrail.general.phi_leakage"

    def test_result_carries_the_registry_key(self) -> None:
        guardrail = PHILeakageGuardrail()
        assert guardrail.check("x").guardrail_key == "guardrail.general.phi_leakage"

    def test_provenance_is_serialisable(self) -> None:
        guardrail = PHILeakageGuardrail()
        assert isinstance(guardrail.provenance().model_dump_json(), str)


def _fake_deid_result(text: str, detections: list[Any]) -> DeidResult:
    return DeidResult(
        text=text,
        report=DeidReport(
            document_id="doc-1",
            entity_counts={},
            detections=detections,
            residual_risk=RiskEstimate(level="low", rationale="test"),
            engine_config_hash="hash",
        ),
    )


class TestCategoryLabelling:
    def test_categories_are_sorted_and_deduplicated(self) -> None:
        from openbtk.core.schemas import TextSpan
        from openbtk.deid.schemas import Detection

        guardrail = PHILeakageGuardrail()
        guardrail._engine = MagicMock(
            deidentify=MagicMock(
                return_value=_fake_deid_result(
                    "x",
                    [
                        Detection(
                            category=PHICategory.SSN,
                            span=TextSpan(start=0, end=1, label="ssn", confidence=0.9),
                            confidence=0.9,
                            method="rule",
                        ),
                        Detection(
                            category=PHICategory.SSN,
                            span=TextSpan(start=2, end=3, label="ssn", confidence=0.9),
                            confidence=0.9,
                            method="rule",
                        ),
                        Detection(
                            category=PHICategory.PHONE_NUMBER,
                            span=TextSpan(
                                start=4, end=5, label="phone", confidence=0.9
                            ),
                            confidence=0.9,
                            method="rule",
                        ),
                    ],
                )
            )
        )
        result = guardrail.check("x")
        assert result.details["categories"] == ["phone_number", "ssn"]
        assert result.details["count"] == 3
