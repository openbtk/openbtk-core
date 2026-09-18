"""Unit tests for openbtk.data.ehr.temporal.TemporalNormalizer."""

from __future__ import annotations

from datetime import UTC, datetime

from openbtk.core.schemas import CodeSystem
from openbtk.data.ehr.schemas import (
    CodedEvent,
    Demographics,
    Encounter,
    Measurement,
    PatientRecord,
)
from openbtk.data.ehr.temporal import TemporalNormalizer

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


class TestAnchoring:
    def test_borrows_encounter_start_when_event_has_no_timestamp(self) -> None:
        patient = _patient(
            encounters=[Encounter(encounter_id="enc-1", start=_T1)],
            conditions=[
                CodedEvent(code="x", system=CodeSystem.SNOMED, encounter_ref="enc-1")
            ],
        )
        result = TemporalNormalizer().process(patient)
        assert result.conditions[0].timestamp == _T1

    def test_does_not_override_an_existing_timestamp(self) -> None:
        patient = _patient(
            encounters=[Encounter(encounter_id="enc-1", start=_T1)],
            conditions=[
                CodedEvent(
                    code="x",
                    system=CodeSystem.SNOMED,
                    encounter_ref="enc-1",
                    timestamp=_T2,
                )
            ],
        )
        result = TemporalNormalizer().process(patient)
        assert result.conditions[0].timestamp == _T2

    def test_no_encounter_ref_stays_none(self) -> None:
        patient = _patient(conditions=[CodedEvent(code="x", system=CodeSystem.SNOMED)])
        result = TemporalNormalizer().process(patient)
        assert result.conditions[0].timestamp is None

    def test_unresolvable_encounter_ref_stays_none(self) -> None:
        patient = _patient(
            conditions=[
                CodedEvent(code="x", system=CodeSystem.SNOMED, encounter_ref="missing")
            ]
        )
        result = TemporalNormalizer().process(patient)
        assert result.conditions[0].timestamp is None

    def test_encounter_with_no_start_leaves_event_undated(self) -> None:
        patient = _patient(
            encounters=[Encounter(encounter_id="enc-1")],
            conditions=[
                CodedEvent(code="x", system=CodeSystem.SNOMED, encounter_ref="enc-1")
            ],
        )
        result = TemporalNormalizer().process(patient)
        assert result.conditions[0].timestamp is None

    def test_disabled_anchoring_leaves_timestamps_untouched(self) -> None:
        patient = _patient(
            encounters=[Encounter(encounter_id="enc-1", start=_T1)],
            conditions=[
                CodedEvent(code="x", system=CodeSystem.SNOMED, encounter_ref="enc-1")
            ],
        )
        result = TemporalNormalizer(anchor_missing_timestamps=False).process(patient)
        assert result.conditions[0].timestamp is None

    def test_anchoring_applies_to_medications_procedures_and_observations(
        self,
    ) -> None:
        patient = _patient(
            encounters=[Encounter(encounter_id="enc-1", start=_T1)],
            medications=[
                CodedEvent(code="x", system=CodeSystem.RXNORM, encounter_ref="enc-1")
            ],
            procedures=[
                CodedEvent(code="x", system=CodeSystem.CPT, encounter_ref="enc-1")
            ],
            observations=[Measurement(code="x", encounter_ref="enc-1")],
        )
        result = TemporalNormalizer().process(patient)
        assert result.medications[0].timestamp == _T1
        assert result.procedures[0].timestamp == _T1
        assert result.observations[0].timestamp == _T1


class TestSorting:
    def test_conditions_are_sorted_chronologically(self) -> None:
        patient = _patient(
            conditions=[
                CodedEvent(code="second", system=CodeSystem.SNOMED, timestamp=_T2),
                CodedEvent(code="first", system=CodeSystem.SNOMED, timestamp=_T1),
            ]
        )
        result = TemporalNormalizer().process(patient)
        assert [c.code for c in result.conditions] == ["first", "second"]

    def test_undated_events_sort_last(self) -> None:
        patient = _patient(
            conditions=[
                CodedEvent(code="undated", system=CodeSystem.SNOMED),
                CodedEvent(code="dated", system=CodeSystem.SNOMED, timestamp=_T1),
            ]
        )
        result = TemporalNormalizer().process(patient)
        assert [c.code for c in result.conditions] == ["dated", "undated"]

    def test_encounters_are_sorted_chronologically(self) -> None:
        patient = _patient(
            encounters=[
                Encounter(encounter_id="second", start=_T2),
                Encounter(encounter_id="first", start=_T1),
            ]
        )
        result = TemporalNormalizer().process(patient)
        assert [e.encounter_id for e in result.encounters] == ["first", "second"]

    def test_undated_encounters_sort_last(self) -> None:
        patient = _patient(
            encounters=[
                Encounter(encounter_id="undated"),
                Encounter(encounter_id="dated", start=_T1),
            ]
        )
        result = TemporalNormalizer().process(patient)
        assert [e.encounter_id for e in result.encounters] == ["dated", "undated"]


class TestReturnValue:
    def test_returns_a_new_patient_record_not_a_mutation(self) -> None:
        patient = _patient()
        result = TemporalNormalizer().process(patient)
        assert result is not patient
        assert result.patient_id == patient.patient_id

    def test_other_fields_are_preserved(self) -> None:
        patient = _patient(patient_id="pt-42")
        result = TemporalNormalizer().process(patient)
        assert result.patient_id == "pt-42"
        assert result.source_system == "fhir-r4"


class TestRegistration:
    def test_registered_under_the_expected_key(self) -> None:
        assert TemporalNormalizer.registry_key == "preprocessor.ehr.temporal"

    def test_provenance_is_serialisable(self) -> None:
        prov = TemporalNormalizer().provenance()
        assert isinstance(prov.model_dump_json(), str)
