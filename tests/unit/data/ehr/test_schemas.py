"""Unit tests for openbtk.data.ehr.schemas."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from openbtk.core.schemas import CodeSystem
from openbtk.data.ehr.schemas import (
    CodedEvent,
    Demographics,
    Encounter,
    Measurement,
    PatientRecord,
)
from openbtk.deid.schemas import DeidStatus


def _patient(**overrides: object) -> PatientRecord:
    defaults: dict[str, object] = {
        "patient_id": "pt-1",
        "demographics": Demographics(),
        "source_system": "fhir-r4",
    }
    defaults.update(overrides)
    return PatientRecord(**defaults)  # type: ignore[arg-type]


class TestDemographics:
    def test_defaults(self) -> None:
        demo = Demographics()
        assert demo.birth_date is None
        assert demo.gender is None
        assert demo.race is None
        assert demo.ethnicity is None
        assert demo.deceased is False
        assert demo.deceased_date is None

    def test_is_frozen(self) -> None:
        demo = Demographics()
        with pytest.raises(ValidationError, match=r"(?i)frozen"):
            demo.gender = "male"  # type: ignore[misc]

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError, match=r"(?i)extra"):
            Demographics(bogus=1)  # type: ignore[call-arg]


class TestEncounter:
    def test_accepts_a_timezone_aware_start(self) -> None:
        aware = datetime(2024, 3, 14, tzinfo=UTC)
        encounter = Encounter(encounter_id="enc-1", start=aware)
        assert encounter.start == aware

    def test_rejects_a_naive_start(self) -> None:
        naive = datetime(2024, 3, 14)  # noqa: DTZ001 -- the point of this test
        with pytest.raises(ValidationError, match="timezone-aware"):
            Encounter(encounter_id="enc-1", start=naive)

    def test_rejects_a_naive_end(self) -> None:
        naive = datetime(2024, 3, 14)  # noqa: DTZ001 -- the point of this test
        with pytest.raises(ValidationError, match="timezone-aware"):
            Encounter(encounter_id="enc-1", end=naive)

    def test_encounter_id_cannot_be_empty(self) -> None:
        with pytest.raises(ValidationError):
            Encounter(encounter_id="")

    def test_round_trips_through_json(self) -> None:
        encounter = Encounter(encounter_id="enc-1", encounter_type="inpatient")
        restored = Encounter.model_validate_json(encounter.model_dump_json())
        assert restored == encounter


class TestCodedEvent:
    def test_defaults(self) -> None:
        event = CodedEvent(code="385093006", system=CodeSystem.SNOMED)
        assert event.display is None
        assert event.timestamp is None
        assert event.encounter_ref is None
        assert event.status is None

    def test_rejects_a_naive_timestamp(self) -> None:
        naive = datetime(2024, 3, 14)  # noqa: DTZ001 -- the point of this test
        with pytest.raises(ValidationError, match="timezone-aware"):
            CodedEvent(code="385093006", system=CodeSystem.SNOMED, timestamp=naive)

    def test_code_cannot_be_empty(self) -> None:
        with pytest.raises(ValidationError):
            CodedEvent(code="", system=CodeSystem.SNOMED)

    def test_system_is_required(self) -> None:
        with pytest.raises(ValidationError):
            CodedEvent(code="385093006")  # type: ignore[call-arg]


class TestMeasurement:
    def test_defaults(self) -> None:
        m = Measurement(code="6690-2")
        assert m.system == CodeSystem.LOINC
        assert m.display is None
        assert m.value is None
        assert m.unit is None
        assert m.reference_range is None

    def test_accepts_a_qualitative_string_value(self) -> None:
        m = Measurement(code="X", value="Positive")
        assert m.value == "Positive"

    def test_accepts_a_numeric_value(self) -> None:
        m = Measurement(code="6690-2", value=14.2)
        assert m.value == 14.2

    def test_rejects_a_naive_timestamp(self) -> None:
        naive = datetime(2024, 3, 14)  # noqa: DTZ001 -- the point of this test
        with pytest.raises(ValidationError, match="timezone-aware"):
            Measurement(code="6690-2", timestamp=naive)

    def test_reference_range_round_trips(self) -> None:
        m = Measurement(code="6690-2", reference_range=(4.5, 11.0))
        restored = Measurement.model_validate_json(m.model_dump_json())
        assert restored.reference_range == (4.5, 11.0)


class TestPatientRecord:
    def test_defaults(self) -> None:
        patient = _patient()
        assert patient.deid_status == DeidStatus.UNKNOWN
        assert patient.encounters == []
        assert patient.conditions == []
        assert patient.medications == []
        assert patient.procedures == []
        assert patient.observations == []
        assert patient.metadata == {}

    def test_is_frozen(self) -> None:
        patient = _patient()
        with pytest.raises(ValidationError, match=r"(?i)frozen"):
            patient.patient_id = "other"  # type: ignore[misc]

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError, match=r"(?i)extra"):
            _patient(bogus=1)

    def test_patient_id_cannot_be_empty(self) -> None:
        with pytest.raises(ValidationError):
            _patient(patient_id="")

    def test_demographics_is_required(self) -> None:
        with pytest.raises(ValidationError):
            PatientRecord(patient_id="pt-1", source_system="fhir-r4")  # type: ignore[call-arg]

    def test_round_trips_through_json(self) -> None:
        patient = _patient(
            conditions=[CodedEvent(code="385093006", system=CodeSystem.SNOMED)]
        )
        restored = PatientRecord.model_validate_json(patient.model_dump_json())
        assert restored == patient
