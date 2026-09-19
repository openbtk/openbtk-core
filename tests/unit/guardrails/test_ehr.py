"""Unit tests for openbtk.guardrails.ehr: EHRCodeValidityGuardrail,
ReferentialIntegrityGuardrail, UnitPlausibilityGuardrail."""

from __future__ import annotations

from datetime import UTC, datetime

from openbtk.core.schemas import CodeSystem, GuardrailSeverity
from openbtk.data.ehr.schemas import (
    CodedEvent,
    Demographics,
    Encounter,
    Measurement,
    PatientRecord,
)
from openbtk.guardrails.ehr import (
    EHRCodeValidityGuardrail,
    ReferentialIntegrityGuardrail,
    UnitPlausibilityGuardrail,
)
from openbtk.guardrails.terminology_validity import TerminologyValidityGuardrail

_T1 = datetime(2024, 3, 14, tzinfo=UTC)
_T2 = datetime(2024, 3, 15, tzinfo=UTC)


def _patient(**overrides: object) -> PatientRecord:
    defaults: dict[str, object] = {
        "patient_id": "pt-1",
        "demographics": Demographics(),
        "source_system": "fhir-r4",
    }
    defaults.update(overrides)
    return PatientRecord(**defaults)  # type: ignore[arg-type]


class TestEHRCodeValidityGuardrail:
    def test_is_a_terminology_validity_guardrail(self) -> None:
        assert issubclass(EHRCodeValidityGuardrail, TerminologyValidityGuardrail)

    def test_registered_under_its_own_key(self) -> None:
        assert EHRCodeValidityGuardrail.registry_key == "guardrail.ehr.code_validity"

    def test_validates_a_whole_patient_record(self) -> None:
        guardrail = EHRCodeValidityGuardrail()
        patient = _patient(
            conditions=[CodedEvent(code="E11.9", system=CodeSystem.ICD10CM)]
        )
        assert guardrail.check(patient).passed is True

    def test_flags_an_invalid_code(self) -> None:
        guardrail = EHRCodeValidityGuardrail()
        patient = _patient(
            conditions=[CodedEvent(code="bogus", system=CodeSystem.ICD10CM)]
        )
        assert guardrail.check(patient).passed is False

    def test_provenance_is_serialisable(self) -> None:
        assert isinstance(
            EHRCodeValidityGuardrail().provenance().model_dump_json(), str
        )


class TestReferentialIntegrityGuardrail:
    def test_unrelated_payload_passes_with_info(self) -> None:
        guardrail = ReferentialIntegrityGuardrail()
        result = guardrail.check("not a patient record")
        assert result.passed is True
        assert result.severity == GuardrailSeverity.INFO

    def test_no_events_passes(self) -> None:
        guardrail = ReferentialIntegrityGuardrail()
        assert guardrail.check(_patient()).passed is True

    def test_event_within_encounter_window_passes(self) -> None:
        patient = _patient(
            encounters=[Encounter(encounter_id="enc-1", start=_T1, end=_T2)],
            conditions=[
                CodedEvent(
                    code="x",
                    system=CodeSystem.SNOMED,
                    encounter_ref="enc-1",
                    timestamp=_T1,
                )
            ],
        )
        assert ReferentialIntegrityGuardrail().check(patient).passed is True

    def test_dangling_encounter_ref_is_a_violation(self) -> None:
        patient = _patient(
            conditions=[
                CodedEvent(code="x", system=CodeSystem.SNOMED, encounter_ref="missing")
            ]
        )
        result = ReferentialIntegrityGuardrail().check(patient)
        assert result.passed is False
        assert result.severity == GuardrailSeverity.BLOCK
        assert "dangling" in result.details["violations"][0]

    def test_event_before_encounter_start_is_a_violation(self) -> None:
        patient = _patient(
            encounters=[Encounter(encounter_id="enc-1", start=_T2)],
            conditions=[
                CodedEvent(
                    code="x",
                    system=CodeSystem.SNOMED,
                    encounter_ref="enc-1",
                    timestamp=_T1,
                )
            ],
        )
        result = ReferentialIntegrityGuardrail().check(patient)
        assert result.passed is False

    def test_event_after_encounter_end_is_a_violation(self) -> None:
        patient = _patient(
            encounters=[Encounter(encounter_id="enc-1", end=_T1)],
            observations=[Measurement(code="x", encounter_ref="enc-1", timestamp=_T2)],
        )
        result = ReferentialIntegrityGuardrail().check(patient)
        assert result.passed is False

    def test_event_with_no_encounter_ref_is_ignored(self) -> None:
        patient = _patient(conditions=[CodedEvent(code="x", system=CodeSystem.SNOMED)])
        assert ReferentialIntegrityGuardrail().check(patient).passed is True

    def test_event_with_no_timestamp_is_not_checked_against_the_window(self) -> None:
        patient = _patient(
            encounters=[Encounter(encounter_id="enc-1", start=_T1, end=_T1)],
            conditions=[
                CodedEvent(code="x", system=CodeSystem.SNOMED, encounter_ref="enc-1")
            ],
        )
        assert ReferentialIntegrityGuardrail().check(patient).passed is True

    def test_provenance_is_serialisable(self) -> None:
        assert isinstance(
            ReferentialIntegrityGuardrail().provenance().model_dump_json(), str
        )


class TestUnitPlausibilityGuardrail:
    def test_unrelated_payload_passes_with_info(self) -> None:
        result = UnitPlausibilityGuardrail().check("not a patient record")
        assert result.passed is True
        assert result.severity == GuardrailSeverity.INFO

    def test_no_observations_passes(self) -> None:
        assert UnitPlausibilityGuardrail().check(_patient()).passed is True

    def test_known_unit_and_plausible_value_passes(self) -> None:
        patient = _patient(
            observations=[Measurement(code="6690-2", value=7.5, unit="10*3/uL")]
        )
        assert UnitPlausibilityGuardrail().check(patient).passed is True

    def test_unrecognised_unit_is_flagged(self) -> None:
        patient = _patient(
            observations=[Measurement(code="6690-2", value=7.5, unit="bogus-unit")]
        )
        result = UnitPlausibilityGuardrail().check(patient)
        assert result.passed is False
        assert result.severity == GuardrailSeverity.WARNING
        assert "unrecognised unit" in result.details["violations"][0]

    def test_implausible_value_is_flagged(self) -> None:
        patient = _patient(
            observations=[Measurement(code="6690-2", value=400000, unit="10*3/uL")]
        )
        result = UnitPlausibilityGuardrail().check(patient)
        assert result.passed is False
        assert "outside" in result.details["violations"][0]

    def test_no_unit_at_all_is_not_flagged_as_unrecognised(self) -> None:
        patient = _patient(observations=[Measurement(code="6690-2", value=7.5)])
        result = UnitPlausibilityGuardrail().check(patient)
        assert result.passed is True

    def test_unknown_loinc_code_skips_the_range_check(self) -> None:
        patient = _patient(
            observations=[Measurement(code="not-a-known-loinc", value=999999)]
        )
        assert UnitPlausibilityGuardrail().check(patient).passed is True

    def test_qualitative_value_skips_the_range_check(self) -> None:
        patient = _patient(
            observations=[Measurement(code="6690-2", value="Positive", unit="%")]
        )
        assert UnitPlausibilityGuardrail().check(patient).passed is True

    def test_provenance_is_serialisable(self) -> None:
        assert isinstance(
            UnitPlausibilityGuardrail().provenance().model_dump_json(), str
        )
