"""Unit tests for openbtk.deid.schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from openbtk.core.schemas import TextSpan
from openbtk.deid.schemas import (
    DeidMode,
    DeidReport,
    DeidResult,
    DeidStatus,
    Detection,
    PHICategory,
    RiskEstimate,
)


def _span(confidence: float = 0.9) -> TextSpan:
    return TextSpan(start=0, end=3, label="probe", confidence=confidence)


def _report(document_id: str = "doc-1") -> DeidReport:
    return DeidReport(
        document_id=document_id,
        entity_counts={},
        detections=[],
        residual_risk=RiskEstimate(level="low", rationale="clean"),
        engine_config_hash="abc123",
    )


class TestPHICategory:
    def test_exactly_eighteen_safe_harbor_categories(self) -> None:
        """45 CFR 164.514(b)(2) enumerates exactly 18 identifier categories."""
        assert len(PHICategory) == 18

    def test_members_are_lowercase_snake_case_values(self) -> None:
        for member in PHICategory:
            assert member.value == member.value.lower()
            assert " " not in member.value


class TestDeidMode:
    def test_exactly_five_modes(self) -> None:
        assert len(DeidMode) == 5


class TestDeidStatus:
    def test_exactly_four_states(self) -> None:
        assert len(DeidStatus) == 4

    def test_unknown_is_the_documented_default_sentinel(self) -> None:
        """docs/05_DATA_MODALITY_SPEC.md section 1.1:
        deid_status: DeidStatus = DeidStatus.UNKNOWN."""
        assert DeidStatus.UNKNOWN.value == "unknown"

    def test_members_are_lowercase(self) -> None:
        for member in DeidStatus:
            assert member.value == member.value.lower()


class TestDetection:
    def test_is_frozen(self) -> None:
        d = Detection(
            category=PHICategory.SSN, span=_span(), confidence=0.9, method="rule"
        )
        with pytest.raises(ValidationError, match=r"(?i)frozen"):
            d.confidence = 0.1  # type: ignore[misc]

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError, match=r"(?i)extra"):
            Detection(
                category=PHICategory.SSN,
                span=_span(),
                confidence=0.9,
                method="rule",
                bogus=1,  # type: ignore[call-arg]
            )

    @pytest.mark.parametrize("confidence", [-0.01, 1.01])
    def test_confidence_out_of_bounds_rejected(self, confidence: float) -> None:
        with pytest.raises(ValidationError):
            Detection(
                category=PHICategory.SSN,
                span=_span(),
                confidence=confidence,
                method="rule",
            )

    def test_rejects_unknown_method(self) -> None:
        with pytest.raises(ValidationError):
            Detection(
                category=PHICategory.SSN,
                span=_span(),
                confidence=0.9,
                method="made_up",
            )

    def test_round_trips_through_json(self) -> None:
        d = Detection(
            category=PHICategory.MEDICAL_RECORD_NUMBER,
            span=_span(),
            confidence=0.75,
            method="ensemble",
        )
        restored = Detection.model_validate_json(d.model_dump_json())
        assert restored == d


class TestDeidReport:
    def test_is_frozen(self) -> None:
        report = _report()
        with pytest.raises(ValidationError, match=r"(?i)frozen"):
            report.document_id = "mutated"  # type: ignore[misc]

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError, match=r"(?i)extra"):
            DeidReport(
                document_id="doc-1",
                entity_counts={},
                detections=[],
                residual_risk=RiskEstimate(level="low", rationale="clean"),
                engine_config_hash="abc123",
                bogus=1,  # type: ignore[call-arg]
            )

    def test_entity_counts_keys_are_real_categories_not_bare_strings(self) -> None:
        report = DeidReport(
            document_id="doc-1",
            entity_counts={PHICategory.SSN: 2},
            detections=[],
            residual_risk=RiskEstimate(level="low", rationale="clean"),
            engine_config_hash="abc123",
        )
        assert isinstance(next(iter(report.entity_counts)), PHICategory)

    def test_round_trips_through_json(self) -> None:
        report = _report()
        restored = DeidReport.model_validate_json(report.model_dump_json())
        assert restored == report

    def test_schema_has_no_field_shaped_to_hold_raw_content(self) -> None:
        """A structural guard, not a behavioural one: DeidReport's own field
        names must never look like a place raw text could be stashed. This
        cannot catch someone MISUSING a legitimate field, but it does catch
        a future field like `raw_text` or `original_value` being added by
        accident."""
        forbidden_substrings = ("text", "content", "value", "raw")
        for name in DeidReport.model_fields:
            lowered = name.lower()
            assert not any(bad in lowered for bad in forbidden_substrings), (
                f"DeidReport field {name!r} looks like it could hold document "
                "content -- see this module's docstring on why that must never "
                "happen."
            )


class TestDeidResult:
    def test_is_frozen(self) -> None:
        result = DeidResult(text="hello", report=_report())
        with pytest.raises(ValidationError, match=r"(?i)frozen"):
            result.text = "mutated"  # type: ignore[misc]

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError, match=r"(?i)extra"):
            DeidResult(text="hello", report=_report(), bogus=1)  # type: ignore[call-arg]
