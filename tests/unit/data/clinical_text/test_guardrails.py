"""Unit tests for opentbtk.data.clinical_text.guardrails."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from opentbtk.data.clinical_text.guardrails import (
    PHIGuardrail,
    HallucinationGuardrail,
    CodeValidityGuardrail,
)
from opentbtk.core.schemas import GuardrailSeverity
from opentbtk.core.errors import GuardrailViolation


class TestPHIGuardrail:
    def test_non_string_payload_passes_with_info(self) -> None:
        guard = PHIGuardrail()
        result = guard.check(12345)
        assert result.passed is True
        assert result.severity == GuardrailSeverity.INFO

    def test_no_phi_detected_passes(self) -> None:
        guard = PHIGuardrail()
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = []
        guard._analyzer = mock_analyzer

        result = guard.check("The lungs are clear bilaterally.")
        assert result.passed is True
        assert result.severity == GuardrailSeverity.INFO

    def test_phi_detected_blocks(self) -> None:
        guard = PHIGuardrail()
        mock_result = MagicMock()
        mock_result.entity_type = "PERSON"
        mock_result.start = 0
        mock_result.end = 10
        mock_result.score = 0.95

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [mock_result]
        guard._analyzer = mock_analyzer

        result = guard.check("John Smith was admitted today.")
        assert result.passed is False
        assert result.severity == GuardrailSeverity.BLOCK
        assert "PERSON" in result.details["entity_types"]

    def test_raise_on_block_raises_violation(self) -> None:
        guard = PHIGuardrail(raise_on_block=True)
        mock_result = MagicMock()
        mock_result.entity_type = "PERSON"
        mock_result.start = 0
        mock_result.end = 10
        mock_result.score = 0.95

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [mock_result]
        guard._analyzer = mock_analyzer

        with pytest.raises(GuardrailViolation):
            guard.check("John Smith was admitted today.")


class TestHallucinationGuardrail:
    def test_high_overlap_passes(self) -> None:
        guard = HallucinationGuardrail(min_overlap_ratio=0.3)
        context = "Patient presents with chest pain and shortness of breath."
        generated = "The patient has chest pain and breath difficulty."

        result = guard.check({"generated": generated, "context": context})
        assert result.passed is True

    def test_low_overlap_warns(self) -> None:
        guard = HallucinationGuardrail(min_overlap_ratio=0.8)
        context = "Patient presents with mild headache."
        generated = "Patient was diagnosed with terminal cancer and given six months to live."

        result = guard.check({"generated": generated, "context": context})
        assert result.passed is False
        assert result.severity == GuardrailSeverity.WARNING

    def test_missing_context_returns_warning_skip(self) -> None:
        guard = HallucinationGuardrail()
        result = guard.check({"generated": "Some text.", "context": ""})
        assert result.passed is True
        assert result.severity == GuardrailSeverity.WARNING

    def test_invalid_payload_format_skips(self) -> None:
        guard = HallucinationGuardrail()
        result = guard.check("not a dict")
        assert result.passed is True
        assert result.severity == GuardrailSeverity.INFO


class TestCodeValidityGuardrail:
    def test_valid_icd10_passes(self) -> None:
        guard = CodeValidityGuardrail()
        result = guard.check("Patient diagnosed with E11.9 type 2 diabetes.")
        assert result.passed is True

    def test_non_string_payload_skips(self) -> None:
        guard = CodeValidityGuardrail()
        result = guard.check(None)
        assert result.passed is True
        assert result.severity == GuardrailSeverity.INFO

    def test_check_can_be_disabled(self) -> None:
        guard = CodeValidityGuardrail(check_icd10=False, check_loinc=False)
        result = guard.check("Anything goes here E99.999999.")
        assert result.passed is True
