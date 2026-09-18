"""Unit tests for openbtk.pipelines.timeline.PatientTimelineSerializer."""

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
from openbtk.deid.schemas import DeidStatus
from openbtk.pipelines.timeline import PatientTimelineSerializer

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


class TestBasicSerialization:
    def test_produces_a_record_id_and_patient_ref(self) -> None:
        record = PatientTimelineSerializer().serialize(_patient(patient_id="pt-42"))
        assert record.record_id == "timeline:pt-42"
        assert record.patient_ref == "pt-42"

    def test_propagates_deid_status_unchanged(self) -> None:
        patient = _patient(deid_status=DeidStatus.DEIDENTIFIED)
        record = PatientTimelineSerializer().serialize(patient)
        assert record.deid_status == DeidStatus.DEIDENTIFIED

    def test_empty_patient_yields_empty_text(self) -> None:
        record = PatientTimelineSerializer().serialize(_patient())
        assert record.text == ""

    def test_note_type_is_ehr_timeline(self) -> None:
        record = PatientTimelineSerializer().serialize(_patient())
        assert record.note_type == "EHR Timeline"

    def test_custom_source(self) -> None:
        record = PatientTimelineSerializer(source="custom-source").serialize(_patient())
        assert record.source == "custom-source"


class TestChronologicalOrdering:
    def test_rows_are_sorted_by_timestamp(self) -> None:
        patient = _patient(
            conditions=[
                CodedEvent(
                    code="second",
                    system=CodeSystem.SNOMED,
                    display="Second",
                    timestamp=_T2,
                ),
                CodedEvent(
                    code="first",
                    system=CodeSystem.SNOMED,
                    display="First",
                    timestamp=_T1,
                ),
            ]
        )
        record = PatientTimelineSerializer().serialize(patient)
        lines = record.text.splitlines()
        assert "First" in lines[0]
        assert "Second" in lines[1]

    def test_undated_rows_sort_last(self) -> None:
        patient = _patient(
            conditions=[
                CodedEvent(code="undated", system=CodeSystem.SNOMED, display="Undated"),
                CodedEvent(
                    code="dated",
                    system=CodeSystem.SNOMED,
                    display="Dated",
                    timestamp=_T1,
                ),
            ]
        )
        record = PatientTimelineSerializer().serialize(patient)
        lines = record.text.splitlines()
        assert "Dated" in lines[0]
        assert "Undated" in lines[1]
        assert "undated" in lines[1].split("|")[0]


class TestRowRendering:
    def test_encounter_line_uses_inpatient_admission_wording(self) -> None:
        patient = _patient(
            encounters=[
                Encounter(encounter_id="e1", encounter_type="inpatient", start=_T1)
            ]
        )
        record = PatientTimelineSerializer().serialize(patient)
        assert "Inpatient admission" in record.text
        assert "2024-03-14" in record.text
        assert "Encounter" in record.text

    def test_ambulatory_encounter_uses_visit_wording(self) -> None:
        patient = _patient(
            encounters=[Encounter(encounter_id="e1", encounter_type="ambulatory")]
        )
        record = PatientTimelineSerializer().serialize(patient)
        assert "Ambulatory visit" in record.text

    def test_unspecified_encounter_type(self) -> None:
        patient = _patient(encounters=[Encounter(encounter_id="e1")])
        record = PatientTimelineSerializer().serialize(patient)
        assert "Unspecified encounter" in record.text

    def test_condition_line_includes_display_and_code(self) -> None:
        patient = _patient(
            conditions=[
                CodedEvent(
                    code="385093006",
                    system=CodeSystem.SNOMED,
                    display="Community-acquired pneumonia",
                    timestamp=_T1,
                )
            ]
        )
        record = PatientTimelineSerializer().serialize(patient)
        assert "Community-acquired pneumonia (SNOMED 385093006)" in record.text
        assert "Condition" in record.text

    def test_condition_without_display_falls_back_to_code(self) -> None:
        patient = _patient(
            conditions=[CodedEvent(code="385093006", system=CodeSystem.SNOMED)]
        )
        record = PatientTimelineSerializer().serialize(patient)
        assert "385093006 (SNOMED 385093006)" in record.text

    def test_medication_line(self) -> None:
        patient = _patient(
            medications=[
                CodedEvent(
                    code="309090",
                    system=CodeSystem.RXNORM,
                    display="Ceftriaxone 1 g IV q24h",
                    timestamp=_T2,
                )
            ]
        )
        record = PatientTimelineSerializer().serialize(patient)
        assert "Ceftriaxone 1 g IV q24h (RXNORM 309090)" in record.text
        assert "Medication" in record.text

    def test_procedure_line(self) -> None:
        patient = _patient(
            procedures=[
                CodedEvent(
                    code="71046",
                    system=CodeSystem.CPT,
                    display="Chest X-ray",
                    timestamp=_T1,
                )
            ]
        )
        record = PatientTimelineSerializer().serialize(patient)
        assert "Chest X-ray (CPT 71046)" in record.text
        assert "Procedure" in record.text

    def test_lab_line_with_value_and_reference_range(self) -> None:
        patient = _patient(
            observations=[
                Measurement(
                    code="6690-2",
                    display="WBC",
                    value=14.2,
                    unit="10*3/uL",
                    reference_range=(4.5, 11.0),
                    timestamp=_T1,
                )
            ]
        )
        record = PatientTimelineSerializer().serialize(patient)
        assert "WBC 14.2 10*3/uL (ref 4.5-11.0) [HIGH]" in record.text
        assert "Lab" in record.text

    def test_lab_line_flags_low_value(self) -> None:
        patient = _patient(
            observations=[
                Measurement(
                    code="x", display="Potassium", value=2.0, reference_range=(3.5, 5.0)
                )
            ]
        )
        record = PatientTimelineSerializer().serialize(patient)
        assert "[LOW]" in record.text

    def test_lab_line_no_flag_within_range(self) -> None:
        patient = _patient(
            observations=[
                Measurement(
                    code="x", display="Potassium", value=4.0, reference_range=(3.5, 5.0)
                )
            ]
        )
        record = PatientTimelineSerializer().serialize(patient)
        assert "[LOW]" not in record.text
        assert "[HIGH]" not in record.text

    def test_lab_line_with_qualitative_value_and_reference_range_no_flag(self) -> None:
        """A non-numeric value with a reference range still renders the
        range, but _range_flag() can't compare a string to bounds -- covers
        that branch directly rather than assuming it never matters."""
        patient = _patient(
            observations=[
                Measurement(
                    code="x",
                    display="Culture",
                    value="Positive",
                    reference_range=(0.0, 1.0),
                )
            ]
        )
        record = PatientTimelineSerializer().serialize(patient)
        assert "[LOW]" not in record.text
        assert "[HIGH]" not in record.text

    def test_lab_line_with_qualitative_value_no_flag(self) -> None:
        patient = _patient(
            observations=[Measurement(code="x", display="Culture", value="Negative")]
        )
        record = PatientTimelineSerializer().serialize(patient)
        assert "Culture Negative" in record.text
        assert "[LOW]" not in record.text
        assert "[HIGH]" not in record.text

    def test_lab_line_with_no_value_at_all(self) -> None:
        patient = _patient(observations=[Measurement(code="x", display="Culture")])
        record = PatientTimelineSerializer().serialize(patient)
        assert "Culture (LOINC x)" in record.text


class TestCustomLabelsAndTemplates:
    def test_custom_labels(self) -> None:
        patient = _patient(
            conditions=[CodedEvent(code="x", system=CodeSystem.SNOMED, display="X")]
        )
        record = PatientTimelineSerializer(condition_label="Dx").serialize(patient)
        assert "Dx" in record.text
        assert "Condition" not in record.text.split("|")[1]

    def test_custom_encounter_template(self) -> None:
        patient = _patient(encounters=[Encounter(encounter_id="e1")])
        record = PatientTimelineSerializer(
            encounter_template=lambda e: f"custom-{e.encounter_id}"
        ).serialize(patient)
        assert "custom-e1" in record.text
