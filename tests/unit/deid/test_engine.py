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
        """ "llm_verifier" is an honest gap (task 2.9) -- constructing an
        engine that requests it must raise immediately, not silently skip
        it and run with a smaller ensemble than asked for."""
        with pytest.raises(RegistryError, match="llm_verifier"):
            DeidEngine(recognizers=["llm_verifier"])

    def test_requesting_ner_lazily_imports_and_succeeds(self) -> None:
        """ "ner" is NOT in _DEFAULT_RECOGNIZERS (it needs a downloaded
        model), but requesting it explicitly must actually work -- this is
        the regression guard for _resolve_recognizer's lazy submodule
        import, independent of whether the real spaCy model is installed
        in this environment (constructing a recognizer does no I/O)."""
        engine = DeidEngine(recognizers=["ner"])
        assert engine is not None


class TestShieldRuleDetectionsFromNer:
    """Unit-level coverage of DeidEngine._shield_rule_detections_from_ner,
    the ADR-0006-named mitigation ("high-precision rules run first and
    their spans are excluded from NER re-examination") that
    tests/accuracy/test_deid_f1_with_ner.py's corpus-level numbers depend
    on. Hand-built Detection objects -- no model required."""

    @staticmethod
    def _det(method: str, category: PHICategory, start: int, end: int) -> Detection:
        return Detection(
            category=category,
            span=TextSpan(start=start, end=end, label=category.value, confidence=0.9),
            confidence=0.9,
            method=method,
        )

    def test_ner_detection_overlapping_a_rule_detection_is_dropped(self) -> None:
        rule_det = self._det("rule", PHICategory.URL, 0, 10)
        ner_det = self._det("ner", PHICategory.NAME, 5, 15)  # overlaps [5, 10)
        result = DeidEngine._shield_rule_detections_from_ner([rule_det, ner_det])
        assert result == [rule_det]

    def test_non_overlapping_ner_detection_survives(self) -> None:
        rule_det = self._det("rule", PHICategory.URL, 0, 10)
        ner_det = self._det("ner", PHICategory.NAME, 20, 30)
        result = DeidEngine._shield_rule_detections_from_ner([rule_det, ner_det])
        assert rule_det in result
        assert ner_det in result

    def test_touching_but_not_overlapping_ner_detection_survives(self) -> None:
        """Half-open [start, end) semantics, same as SpanMerger: a rule
        span ending at 10 and an NER span starting at 10 share no
        character."""
        rule_det = self._det("rule", PHICategory.URL, 0, 10)
        ner_det = self._det("ner", PHICategory.NAME, 10, 20)
        result = DeidEngine._shield_rule_detections_from_ner([rule_det, ner_det])
        assert ner_det in result

    def test_two_overlapping_ner_detections_are_both_kept(self) -> None:
        """The shield only ever removes "ner" detections that overlap a
        "rule" one -- ner-vs-ner overlap is SpanMerger's job entirely,
        unaffected by this pre-filter."""
        ner_a = self._det("ner", PHICategory.NAME, 0, 10)
        ner_b = self._det("ner", PHICategory.GEOGRAPHIC_SUBDIVISION, 5, 15)
        result = DeidEngine._shield_rule_detections_from_ner([ner_a, ner_b])
        assert result == [ner_a, ner_b]

    def test_no_rule_detections_at_all_is_a_no_op(self) -> None:
        ner_det = self._det("ner", PHICategory.NAME, 0, 10)
        result = DeidEngine._shield_rule_detections_from_ner([ner_det])
        assert result == [ner_det]

    def test_empty_input_returns_empty_output(self) -> None:
        assert DeidEngine._shield_rule_detections_from_ner([]) == []
