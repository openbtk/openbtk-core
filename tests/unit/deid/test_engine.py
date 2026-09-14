"""Unit tests for openbtk.deid.engine.DeidEngine.

Registers a small, self-contained, fixed-confidence-0.5 recognizer locally
(same self-registration pattern as tests/unit/core/test_config.py and
tests/security/test_error_context.py -- not tests/contract/conftest.py's
ReferenceRecognizer, which is only registered once pytest loads that
conftest; this file must also pass when run standalone) to test
recall_bias filtering directly -- RuleRecognizer always reports 0.95,
which never distinguishes "high"/"balanced"/"precision" from each other.
"""

from __future__ import annotations

import re
from typing import ClassVar, Literal

import pytest

from openbtk.core.errors import DeidError, RegistryError
from openbtk.core.schemas import TextSpan
from openbtk.deid.engine import DeidEngine
from openbtk.deid.recognizers.base import RECOGNIZER_REGISTRY, BaseRecognizer
from openbtk.deid.schemas import DeidMode, Detection, PHICategory

_PROBE_KEY = "recognizer.general.engine_test_probe"

if not RECOGNIZER_REGISTRY.is_registered(_PROBE_KEY):

    @RECOGNIZER_REGISTRY.register(_PROBE_KEY)
    class _LowConfidenceProbeRecognizer(BaseRecognizer):
        """Detects any run of digits at a fixed, low confidence (0.5) --
        exists purely to exercise recall_bias's threshold boundaries."""

        method: ClassVar[Literal["rule", "ner", "llm_verifier"]] = "rule"
        _PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"\d+")

        def detect(self, text: str) -> list[Detection]:
            return [
                Detection(
                    category=PHICategory.OTHER_UNIQUE_IDENTIFIER,
                    span=TextSpan(
                        start=m.start(), end=m.end(), label="digits", confidence=0.5
                    ),
                    confidence=0.5,
                    method=self.method,
                )
                for m in self._PATTERN.finditer(text)
            ]


_VERY_LOW_PROBE_KEY = "recognizer.general.engine_test_very_low_probe"

if not RECOGNIZER_REGISTRY.is_registered(_VERY_LOW_PROBE_KEY):

    @RECOGNIZER_REGISTRY.register(_VERY_LOW_PROBE_KEY)
    class _VeryLowConfidenceProbeRecognizer(BaseRecognizer):
        """Same shape as _LowConfidenceProbeRecognizer, at confidence 0.3 --
        exists to reach residual-risk's "high" bucket (< 0.5), which no
        other fixture in this file produces."""

        method: ClassVar[Literal["rule", "ner", "llm_verifier"]] = "rule"
        _PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"\d+")

        def detect(self, text: str) -> list[Detection]:
            return [
                Detection(
                    category=PHICategory.OTHER_UNIQUE_IDENTIFIER,
                    span=TextSpan(
                        start=m.start(), end=m.end(), label="digits", confidence=0.3
                    ),
                    confidence=0.3,
                    method=self.method,
                )
                for m in self._PATTERN.finditer(text)
            ]


class TestEndToEnd:
    def test_ssn_is_removed_from_the_output_text(self) -> None:
        engine = DeidEngine(mode=DeidMode.REDACT, recognizers=["rule"])
        text = "SSN: 123-45-6789."  # phi-fixture-ok
        result = engine.deidentify(text, patient_id="p1")
        assert "123-45-6789" not in result.text  # phi-fixture-ok
        assert "[REDACTED]" in result.text

    def test_surrogate_mode_is_stable_across_two_calls_for_the_same_patient(
        self,
    ) -> None:
        engine = DeidEngine(mode=DeidMode.SURROGATE, recognizers=["rule"])
        text = "MRN: MRN-4821093."  # phi-fixture-ok
        first = engine.deidentify(text, patient_id="p1")
        second = engine.deidentify(text, patient_id="p1")
        assert first.text == second.text

    def test_report_document_id_is_auto_generated_when_not_given(self) -> None:
        engine = DeidEngine(mode=DeidMode.REDACT, recognizers=["rule"])
        result = engine.deidentify("no phi here", patient_id="p1")
        assert result.report.document_id  # non-empty

    def test_report_document_id_uses_the_caller_supplied_value(self) -> None:
        engine = DeidEngine(mode=DeidMode.REDACT, recognizers=["rule"])
        result = engine.deidentify(
            "no phi here", patient_id="p1", document_id="my-doc-42"
        )
        assert result.report.document_id == "my-doc-42"

    def test_entity_counts_match_the_surviving_detections(self) -> None:
        engine = DeidEngine(mode=DeidMode.REDACT, recognizers=["rule"])
        text = "SSN: 123-45-6789. Second SSN: 987-65-4321."  # phi-fixture-ok
        result = engine.deidentify(text, patient_id="p1")
        assert result.report.entity_counts[PHICategory.SSN] == 2

    def test_report_carries_no_raw_text_field_at_all(self) -> None:
        """Structural: DeidReport simply has no field capable of holding
        the original text -- see openbtk.deid.schemas' own test for this
        same property; here it's exercised end to end through a real engine
        run rather than an empty report constructed by hand."""
        engine = DeidEngine(mode=DeidMode.REDACT, recognizers=["rule"])
        ssn = "123-45-6789"  # phi-fixture-ok
        result = engine.deidentify(f"SSN: {ssn}.", patient_id="p1")
        assert "text" not in type(result.report).model_fields
        assert ssn not in result.report.model_dump_json()


class TestRecallBiasFiltering:
    def test_high_recall_bias_keeps_low_confidence_detections(self) -> None:
        engine = DeidEngine(
            mode=DeidMode.REDACT,
            recall_bias="high",
            recognizers=["engine_test_probe"],
        )
        result = engine.deidentify("Ref: 98765", patient_id="p1")
        assert "98765" not in result.text

    def test_precision_recall_bias_drops_low_confidence_detections(self) -> None:
        """The probe recognizer reports confidence 0.5; "precision"
        requires >= 0.8, so its detections must not survive."""
        engine = DeidEngine(
            mode=DeidMode.REDACT,
            recall_bias="precision",
            recognizers=["engine_test_probe"],
        )
        result = engine.deidentify("Ref: 98765", patient_id="p1")
        assert "98765" in result.text  # NOT redacted -- filtered out entirely
        assert result.report.entity_counts == {}

    def test_balanced_recall_bias_keeps_exactly_threshold_confidence(self) -> None:
        """balanced's threshold is 0.5; the probe recognizer's confidence
        IS 0.5 -- the boundary must be inclusive (>=), not exclusive (>)."""
        engine = DeidEngine(
            mode=DeidMode.REDACT,
            recall_bias="balanced",
            recognizers=["engine_test_probe"],
        )
        result = engine.deidentify("Ref: 98765", patient_id="p1")
        assert "98765" not in result.text

    def test_invalid_recall_bias_raises_deid_error(self) -> None:
        with pytest.raises(DeidError):
            DeidEngine(recall_bias="nonexistent")  # type: ignore[arg-type]


class TestResidualRisk:
    def test_zero_detections_reports_medium_risk_with_an_honest_rationale(self) -> None:
        engine = DeidEngine(mode=DeidMode.REDACT, recognizers=["rule"])
        result = engine.deidentify("nothing sensitive here at all", patient_id="p1")
        assert result.report.residual_risk.level == "medium"
        assert "does not guarantee" in result.report.residual_risk.rationale

    def test_high_confidence_detections_report_low_risk(self) -> None:
        # RuleRecognizer always reports 0.95.
        engine = DeidEngine(mode=DeidMode.REDACT, recognizers=["rule"])
        ssn = "123-45-6789"  # phi-fixture-ok
        result = engine.deidentify(f"SSN: {ssn}.", patient_id="p1")
        assert result.report.residual_risk.level == "low"

    def test_low_confidence_survivors_report_higher_risk(self) -> None:
        engine = DeidEngine(
            mode=DeidMode.REDACT,
            recall_bias="high",
            recognizers=["engine_test_probe"],
        )
        result = engine.deidentify("Ref: 98765", patient_id="p1")
        assert result.report.residual_risk.level == "medium"  # confidence 0.5

    def test_very_low_confidence_survivors_report_high_risk(self) -> None:
        engine = DeidEngine(
            mode=DeidMode.REDACT,
            recall_bias="high",
            recognizers=["engine_test_very_low_probe"],
        )
        result = engine.deidentify("Ref: 98765", patient_id="p1")
        assert result.report.residual_risk.level == "high"  # confidence 0.3


class TestConfigHash:
    def test_same_configuration_produces_the_same_hash(self) -> None:
        a = DeidEngine(mode=DeidMode.REDACT, recall_bias="high", recognizers=["rule"])
        b = DeidEngine(mode=DeidMode.REDACT, recall_bias="high", recognizers=["rule"])
        result_a = a.deidentify("hello", patient_id="p1")
        result_b = b.deidentify("hello", patient_id="p1")
        assert result_a.report.engine_config_hash == result_b.report.engine_config_hash

    def test_different_mode_produces_a_different_hash(self) -> None:
        a = DeidEngine(mode=DeidMode.REDACT, recognizers=["rule"])
        b = DeidEngine(mode=DeidMode.TAG, recognizers=["rule"])
        result_a = a.deidentify("hello", patient_id="p1")
        result_b = b.deidentify("hello", patient_id="p1")
        assert result_a.report.engine_config_hash != result_b.report.engine_config_hash


class TestUnavailableRecognizer:
    def test_requesting_an_unbuilt_recognizer_fails_loudly_not_silently(self) -> None:
        """ "ner" and "llm_verifier" are honest gaps at M2 (tasks 2.4/2.9) --
        constructing an engine that requests one must raise immediately,
        not silently skip it and run with a smaller ensemble than asked
        for."""
        with pytest.raises(RegistryError, match="ner"):
            DeidEngine(recognizers=["ner"])
