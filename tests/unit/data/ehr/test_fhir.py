"""Unit tests for openbtk.data.ehr.fhir.FHIRLoader.

Every test drives the loader through real ``fhir.resources`` R4B model
validation -- no mocking of the FHIR library itself, the same "real,
installed library, no mocking" philosophy already used for
tests/unit/retrieval's vector store tests (task 5.6), since fhir.resources
is free/fast/local, not a network- or cost-bearing dependency.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from openbtk.core.errors import LoaderError
from openbtk.core.schemas import CodeSystem
from openbtk.data.ehr.fhir import FHIRLoader, _reference_id

if TYPE_CHECKING:
    from pathlib import Path

# The 'ehr' extra (fhir.resources) is genuinely optional; skip this whole
# module in CI's zero-extras test-core job instead of failing at load().
pytest.importorskip("fhir.resources")


def _bundle(*resources: dict[str, Any]) -> dict[str, Any]:
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [{"resource": r} for r in resources],
    }


def _patient(**overrides: Any) -> dict[str, Any]:
    resource = {"resourceType": "Patient", "id": "pt-1"}
    resource.update(overrides)
    return resource


def _encounter(**overrides: Any) -> dict[str, Any]:
    resource: dict[str, Any] = {
        "resourceType": "Encounter",
        "id": "enc-1",
        "status": "finished",
        "class": {"code": "AMB"},
        "subject": {"reference": "Patient/pt-1"},
    }
    resource.update(overrides)
    return resource


def _condition(**overrides: Any) -> dict[str, Any]:
    resource: dict[str, Any] = {
        "resourceType": "Condition",
        "id": "cond-1",
        "subject": {"reference": "Patient/pt-1"},
        "encounter": {"reference": "Encounter/enc-1"},
        "code": {
            "coding": [
                {
                    "system": "http://snomed.info/sct",
                    "code": "385093006",
                    "display": "Community-acquired pneumonia",
                }
            ]
        },
    }
    resource.update(overrides)
    return resource


def _medication_request(**overrides: Any) -> dict[str, Any]:
    resource: dict[str, Any] = {
        "resourceType": "MedicationRequest",
        "id": "med-1",
        "status": "active",
        "intent": "order",
        "subject": {"reference": "Patient/pt-1"},
        "medicationCodeableConcept": {
            "coding": [
                {
                    "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                    "code": "309090",
                    "display": "Ceftriaxone 1 g Injection",
                }
            ]
        },
    }
    resource.update(overrides)
    return resource


def _medication_statement(**overrides: Any) -> dict[str, Any]:
    resource: dict[str, Any] = {
        "resourceType": "MedicationStatement",
        "id": "medstmt-1",
        "status": "active",
        "subject": {"reference": "Patient/pt-1"},
        "medicationCodeableConcept": {
            "coding": [
                {
                    "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                    "code": "309090",
                }
            ]
        },
    }
    resource.update(overrides)
    return resource


def _procedure(**overrides: Any) -> dict[str, Any]:
    resource: dict[str, Any] = {
        "resourceType": "Procedure",
        "id": "proc-1",
        "status": "completed",
        "subject": {"reference": "Patient/pt-1"},
        "code": {
            "coding": [
                {
                    "system": "http://www.ama-assn.org/go/cpt",
                    "code": "71046",
                    "display": "Chest X-ray",
                }
            ]
        },
    }
    resource.update(overrides)
    return resource


def _observation(**overrides: Any) -> dict[str, Any]:
    resource: dict[str, Any] = {
        "resourceType": "Observation",
        "id": "obs-1",
        "status": "final",
        "subject": {"reference": "Patient/pt-1"},
        "code": {
            "coding": [
                {"system": "http://loinc.org", "code": "6690-2", "display": "WBC"}
            ]
        },
        "valueQuantity": {"value": 14.2, "unit": "10*3/uL"},
    }
    resource.update(overrides)
    return resource


def _write_bundle(directory: Path, filename: str, bundle: dict[str, Any]) -> None:
    (directory / filename).write_text(json.dumps(bundle), encoding="utf-8")


class TestReferenceId:
    def test_strips_a_resource_type_prefix(self) -> None:
        assert _reference_id("Encounter/enc-1") == "enc-1"

    def test_strips_a_urn_uuid_prefix(self) -> None:
        assert _reference_id("urn:uuid:abc-123") == "abc-123"

    def test_bare_id_passes_through(self) -> None:
        assert _reference_id("enc-1") == "enc-1"

    def test_none_passes_through(self) -> None:
        assert _reference_id(None) is None


class TestFullBundle:
    def test_all_six_resource_types_map_correctly(self, tmp_path: Path) -> None:
        _write_bundle(
            tmp_path,
            "pt-1.json",
            _bundle(
                _patient(gender="female", birthDate="1980-05-01"),
                _encounter(),
                _condition(),
                _medication_request(),
                _procedure(),
                _observation(),
            ),
        )
        records = list(FHIRLoader().load(str(tmp_path)))
        assert len(records) == 1
        record = records[0]
        assert record.patient_id == "pt-1"
        assert record.source_system == "fhir-r4"
        assert record.demographics.gender == "female"
        assert len(record.encounters) == 1
        assert record.encounters[0].encounter_type == "AMB"
        assert len(record.conditions) == 1
        assert record.conditions[0].system == CodeSystem.SNOMED
        assert record.conditions[0].encounter_ref == "enc-1"
        assert len(record.medications) == 1
        assert record.medications[0].system == CodeSystem.RXNORM
        assert len(record.procedures) == 1
        assert record.procedures[0].system == CodeSystem.CPT
        assert len(record.observations) == 1
        assert record.observations[0].value == 14.2
        assert record.observations[0].reference_range is None

    def test_medication_statement_is_also_recognised(self, tmp_path: Path) -> None:
        _write_bundle(
            tmp_path, "pt-1.json", _bundle(_patient(), _medication_statement())
        )
        records = list(FHIRLoader().load(str(tmp_path)))
        assert len(records[0].medications) == 1

    def test_multiple_bundle_files_yield_multiple_records_sorted(
        self, tmp_path: Path
    ) -> None:
        _write_bundle(tmp_path, "b.json", _bundle(_patient(id="pt-b")))
        _write_bundle(tmp_path, "a.json", _bundle(_patient(id="pt-a")))
        records = list(FHIRLoader().load(str(tmp_path)))
        assert [r.patient_id for r in records] == ["pt-a", "pt-b"]

    def test_metadata_records_the_source_filename(self, tmp_path: Path) -> None:
        _write_bundle(tmp_path, "pt-1.json", _bundle(_patient()))
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.metadata == {"bundle_file": "pt-1.json"}


class TestRaceEthnicityExtensions:
    _RACE_URL = "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race"
    _ETHNICITY_URL = "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity"

    def test_reads_the_text_sub_extension(self, tmp_path: Path) -> None:
        patient = _patient(
            extension=[
                {
                    "url": self._RACE_URL,
                    "extension": [{"url": "text", "valueString": "White"}],
                }
            ]
        )
        _write_bundle(tmp_path, "pt-1.json", _bundle(patient))
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.demographics.race == "White"

    def test_falls_back_to_the_ombcategory_display(self, tmp_path: Path) -> None:
        patient = _patient(
            extension=[
                {
                    "url": self._ETHNICITY_URL,
                    "extension": [
                        {
                            "url": "ombCategory",
                            "valueCoding": {"code": "2135-2", "display": "Hispanic"},
                        }
                    ],
                }
            ]
        )
        _write_bundle(tmp_path, "pt-1.json", _bundle(patient))
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.demographics.ethnicity == "Hispanic"

    def test_absent_extension_leaves_both_none(self, tmp_path: Path) -> None:
        _write_bundle(tmp_path, "pt-1.json", _bundle(_patient()))
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.demographics.race is None
        assert record.demographics.ethnicity is None

    def test_matching_extension_with_neither_sub_extension_leaves_none(
        self, tmp_path: Path
    ) -> None:
        patient = _patient(
            extension=[
                {
                    "url": self._RACE_URL,
                    "extension": [{"url": "detailed", "valueString": "irrelevant"}],
                }
            ]
        )
        _write_bundle(tmp_path, "pt-1.json", _bundle(patient))
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.demographics.race is None


class TestDeceased:
    def test_deceased_boolean_true(self, tmp_path: Path) -> None:
        _write_bundle(tmp_path, "pt-1.json", _bundle(_patient(deceasedBoolean=True)))
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.demographics.deceased is True
        assert record.demographics.deceased_date is None

    def test_deceased_datetime_sets_both_fields(self, tmp_path: Path) -> None:
        _write_bundle(
            tmp_path,
            "pt-1.json",
            _bundle(_patient(deceasedDateTime="2024-06-01T00:00:00Z")),
        )
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.demographics.deceased is True
        assert record.demographics.deceased_date is not None
        assert record.demographics.deceased_date.isoformat() == "2024-06-01"

    def test_not_deceased_by_default(self, tmp_path: Path) -> None:
        _write_bundle(tmp_path, "pt-1.json", _bundle(_patient()))
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.demographics.deceased is False


class TestTimestampFallbacks:
    def test_condition_falls_back_to_onset_period(self, tmp_path: Path) -> None:
        cond = _condition(onsetPeriod={"start": "2024-03-14T08:00:00Z"})
        _write_bundle(tmp_path, "pt-1.json", _bundle(_patient(), cond))
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.conditions[0].timestamp is not None

    def test_condition_falls_back_to_recorded_date(self, tmp_path: Path) -> None:
        cond = _condition(recordedDate="2024-03-14T08:00:00Z")
        _write_bundle(tmp_path, "pt-1.json", _bundle(_patient(), cond))
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.conditions[0].timestamp is not None

    def test_condition_status_from_clinical_status(self, tmp_path: Path) -> None:
        cond = _condition(
            clinicalStatus={"coding": [{"code": "active"}]},
        )
        _write_bundle(tmp_path, "pt-1.json", _bundle(_patient(), cond))
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.conditions[0].status == "active"

    def test_medication_statement_falls_back_to_effective_period(
        self, tmp_path: Path
    ) -> None:
        ms = _medication_statement(effectivePeriod={"start": "2024-03-14T08:00:00Z"})
        _write_bundle(tmp_path, "pt-1.json", _bundle(_patient(), ms))
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.medications[0].timestamp is not None

    def test_medication_statement_encounter_ref_from_context(
        self, tmp_path: Path
    ) -> None:
        ms = _medication_statement(context={"reference": "Encounter/enc-1"})
        _write_bundle(tmp_path, "pt-1.json", _bundle(_patient(), ms))
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.medications[0].encounter_ref == "enc-1"

    def test_procedure_falls_back_to_performed_period(self, tmp_path: Path) -> None:
        proc = _procedure(performedPeriod={"start": "2024-03-14T08:00:00Z"})
        _write_bundle(tmp_path, "pt-1.json", _bundle(_patient(), proc))
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.procedures[0].timestamp is not None

    def test_observation_falls_back_to_effective_period(self, tmp_path: Path) -> None:
        obs = _observation(effectivePeriod={"start": "2024-03-14T08:00:00Z"})
        obs.pop("valueQuantity")
        _write_bundle(tmp_path, "pt-1.json", _bundle(_patient(), obs))
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.observations[0].timestamp is not None


class TestObservationValueVariants:
    def test_value_string(self, tmp_path: Path) -> None:
        obs = _observation()
        obs.pop("valueQuantity")
        obs["valueString"] = "Positive"
        _write_bundle(tmp_path, "pt-1.json", _bundle(_patient(), obs))
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.observations[0].value == "Positive"

    def test_value_codeable_concept(self, tmp_path: Path) -> None:
        obs = _observation()
        obs.pop("valueQuantity")
        obs["valueCodeableConcept"] = {"coding": [{"code": "x", "display": "Detected"}]}
        _write_bundle(tmp_path, "pt-1.json", _bundle(_patient(), obs))
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.observations[0].value == "Detected"

    def test_no_value_at_all(self, tmp_path: Path) -> None:
        obs = _observation()
        obs.pop("valueQuantity")
        _write_bundle(tmp_path, "pt-1.json", _bundle(_patient(), obs))
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.observations[0].value is None

    def test_reference_range_with_low_and_high(self, tmp_path: Path) -> None:
        obs = _observation(
            referenceRange=[{"low": {"value": 4.5}, "high": {"value": 11.0}}]
        )
        _write_bundle(tmp_path, "pt-1.json", _bundle(_patient(), obs))
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.observations[0].reference_range == (4.5, 11.0)

    def test_reference_range_missing_low_is_ignored(self, tmp_path: Path) -> None:
        obs = _observation(referenceRange=[{"high": {"value": 11.0}}])
        _write_bundle(tmp_path, "pt-1.json", _bundle(_patient(), obs))
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.observations[0].reference_range is None

    def test_encounter_ref_populated(self, tmp_path: Path) -> None:
        obs = _observation(encounter={"reference": "Encounter/enc-1"})
        _write_bundle(tmp_path, "pt-1.json", _bundle(_patient(), obs))
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.observations[0].encounter_ref == "enc-1"


class TestUnrecognisedCodeSystemsAreSkippedNotFatal:
    def test_unrecognised_condition_coding_is_dropped(self, tmp_path: Path) -> None:
        cond = _condition(
            code={"coding": [{"system": "http://example.org/local", "code": "x"}]}
        )
        _write_bundle(tmp_path, "pt-1.json", _bundle(_patient(), cond))
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.conditions == []

    def test_unrecognised_medication_coding_is_dropped(self, tmp_path: Path) -> None:
        mr = _medication_request(
            medicationCodeableConcept={
                "coding": [{"system": "http://example.org/local", "code": "x"}]
            }
        )
        _write_bundle(tmp_path, "pt-1.json", _bundle(_patient(), mr))
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.medications == []

    def test_unrecognised_procedure_coding_is_dropped(self, tmp_path: Path) -> None:
        proc = _procedure(
            code={"coding": [{"system": "http://example.org/local", "code": "x"}]}
        )
        _write_bundle(tmp_path, "pt-1.json", _bundle(_patient(), proc))
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.procedures == []

    def test_unrecognised_observation_coding_is_dropped(self, tmp_path: Path) -> None:
        obs = _observation(
            code={"coding": [{"system": "http://example.org/local", "code": "x"}]}
        )
        _write_bundle(tmp_path, "pt-1.json", _bundle(_patient(), obs))
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.observations == []

    def test_condition_with_no_coding_at_all_is_dropped(self, tmp_path: Path) -> None:
        cond = _condition(code={"coding": []})
        _write_bundle(tmp_path, "pt-1.json", _bundle(_patient(), cond))
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.conditions == []

    def test_unrecognised_medication_statement_coding_is_dropped(
        self, tmp_path: Path
    ) -> None:
        ms = _medication_statement(
            medicationCodeableConcept={
                "coding": [{"system": "http://example.org/local", "code": "x"}]
            }
        )
        _write_bundle(tmp_path, "pt-1.json", _bundle(_patient(), ms))
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.medications == []


class TestEncounterEdgeCases:
    def test_encounter_missing_id_is_skipped(self, tmp_path: Path) -> None:
        enc = _encounter()
        enc.pop("id")
        _write_bundle(tmp_path, "pt-1.json", _bundle(_patient(), enc))
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.encounters == []

    def test_encounter_without_period_has_no_start_or_end(self, tmp_path: Path) -> None:
        _write_bundle(tmp_path, "pt-1.json", _bundle(_patient(), _encounter()))
        record = next(FHIRLoader().load(str(tmp_path)))
        assert record.encounters[0].start is None
        assert record.encounters[0].end is None


class TestBundleEntryEdgeCases:
    def test_an_entry_with_no_resource_is_skipped(self, tmp_path: Path) -> None:
        bundle = _bundle(_patient())
        bundle["entry"].append({})  # a real, valid-but-empty BundleEntry
        _write_bundle(tmp_path, "pt-1.json", bundle)
        records = list(FHIRLoader().load(str(tmp_path)))
        assert len(records) == 1

    def test_an_unhandled_resource_type_is_ignored(self, tmp_path: Path) -> None:
        organization = {"resourceType": "Organization", "id": "org-1"}
        _write_bundle(tmp_path, "pt-1.json", _bundle(_patient(), organization))
        records = list(FHIRLoader().load(str(tmp_path)))
        assert len(records) == 1
        assert records[0].patient_id == "pt-1"


class TestErrorHandling:
    def test_raises_on_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(LoaderError):
            list(FHIRLoader().load(str(tmp_path / "does-not-exist")))

    def test_raises_on_invalid_json(self, tmp_path: Path) -> None:
        (tmp_path / "bad.json").write_text("not json{{{", encoding="utf-8")
        with pytest.raises(LoaderError):
            list(FHIRLoader().load(str(tmp_path)))

    def test_raises_when_resource_type_is_not_bundle(self, tmp_path: Path) -> None:
        (tmp_path / "bad.json").write_text(json.dumps(_patient()), encoding="utf-8")
        with pytest.raises(LoaderError, match="not a FHIR Bundle"):
            list(FHIRLoader().load(str(tmp_path)))

    def test_raises_on_a_structurally_invalid_bundle(self, tmp_path: Path) -> None:
        bad_bundle = {"resourceType": "Bundle", "type": "collection", "entry": "nope"}
        (tmp_path / "bad.json").write_text(json.dumps(bad_bundle), encoding="utf-8")
        with pytest.raises(LoaderError, match="not a valid FHIR R4 Bundle"):
            list(FHIRLoader().load(str(tmp_path)))

    def test_raises_when_bundle_has_no_patient(self, tmp_path: Path) -> None:
        _write_bundle(tmp_path, "pt-1.json", _bundle(_encounter()))
        with pytest.raises(LoaderError, match="contains no Patient"):
            list(FHIRLoader().load(str(tmp_path)))


class TestRegistration:
    def test_registered_under_the_expected_key(self) -> None:
        assert FHIRLoader.registry_key == "loader.ehr.fhir"

    def test_provenance_is_serialisable(self) -> None:
        prov = FHIRLoader().provenance()
        assert isinstance(prov.model_dump_json(), str)
