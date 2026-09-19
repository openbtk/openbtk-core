"""Unit tests for openbtk.data.ehr.omop.OMOPLoader.

Every test drives the loader against real Parquet files written via a real,
installed pyarrow -- no mocking of the library itself (the same "real,
installed, free/local dependency" philosophy as tests/unit/data/ehr/test_fhir.py).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

import pytest

from openbtk.core.errors import LoaderError
from openbtk.core.schemas import CodeSystem
from openbtk.data.ehr.omop import OMOPLoader

if TYPE_CHECKING:
    from pathlib import Path

# The 'ehr' extra (pyarrow) is genuinely optional -- CI's test-core job installs
# zero extras (NFR-10), so this whole module skips there rather than erroring at
# collection. Found by reproducing a clean `pip install -e .[dev]` venv.
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _write_table(directory: Path, filename: str, columns: dict[str, list[Any]]) -> None:
    pq.write_table(pa.table(columns), directory / filename)


def _write_person(
    directory: Path,
    *,
    person_id: list[Any] = [1],  # noqa: B006 -- test-only, never mutated
    gender_concept_id: list[Any] | None = None,
    year_of_birth: list[Any] | None = None,
    month_of_birth: list[Any] | None = None,
    day_of_birth: list[Any] | None = None,
    birth_datetime: list[Any] | None = None,
    race_source_value: list[Any] | None = None,
    ethnicity_source_value: list[Any] | None = None,
) -> None:
    n = len(person_id)
    columns = {
        "person_id": person_id,
        "gender_concept_id": gender_concept_id or [None] * n,
        "year_of_birth": year_of_birth or [None] * n,
        "month_of_birth": month_of_birth or [None] * n,
        "day_of_birth": day_of_birth or [None] * n,
    }
    if birth_datetime is not None:
        columns["birth_datetime"] = birth_datetime
    if race_source_value is not None:
        columns["race_source_value"] = race_source_value
    if ethnicity_source_value is not None:
        columns["ethnicity_source_value"] = ethnicity_source_value
    _write_table(directory, "person.parquet", columns)


class TestBasicLoad:
    def test_raises_on_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(LoaderError):
            list(OMOPLoader().load(str(tmp_path / "nope")))

    def test_raises_when_person_table_is_missing(self, tmp_path: Path) -> None:
        with pytest.raises(LoaderError, match=r"person\.parquet"):
            list(OMOPLoader().load(str(tmp_path)))

    def test_raises_on_a_corrupt_person_table(self, tmp_path: Path) -> None:
        (tmp_path / "person.parquet").write_text("not parquet", encoding="utf-8")
        with pytest.raises(LoaderError):
            list(OMOPLoader().load(str(tmp_path)))

    def test_person_only_directory_yields_bare_records(self, tmp_path: Path) -> None:
        _write_person(tmp_path, person_id=[1, 2])
        records = list(OMOPLoader().load(str(tmp_path)))
        assert [r.patient_id for r in records] == ["1", "2"]
        assert records[0].encounters == []
        assert records[0].conditions == []
        assert records[0].source_system == "omop-cdm-5.4"


class TestDemographics:
    def test_gender_female(self, tmp_path: Path) -> None:
        _write_person(tmp_path, gender_concept_id=[8532])
        record = next(OMOPLoader().load(str(tmp_path)))
        assert record.demographics.gender == "female"

    def test_gender_male(self, tmp_path: Path) -> None:
        _write_person(tmp_path, gender_concept_id=[8507])
        record = next(OMOPLoader().load(str(tmp_path)))
        assert record.demographics.gender == "male"

    def test_unknown_gender_concept_is_none(self, tmp_path: Path) -> None:
        _write_person(tmp_path, gender_concept_id=[9999999])
        record = next(OMOPLoader().load(str(tmp_path)))
        assert record.demographics.gender is None

    def test_no_gender_concept_is_none(self, tmp_path: Path) -> None:
        _write_person(tmp_path)
        record = next(OMOPLoader().load(str(tmp_path)))
        assert record.demographics.gender is None

    def test_birth_date_from_birth_datetime_column(self, tmp_path: Path) -> None:
        _write_person(tmp_path, birth_datetime=[datetime(1980, 5, 1, 3, tzinfo=UTC)])
        record = next(OMOPLoader().load(str(tmp_path)))
        assert record.demographics.birth_date == date(1980, 5, 1)

    def test_birth_date_from_date_typed_birth_datetime_column(
        self, tmp_path: Path
    ) -> None:
        _write_person(tmp_path, birth_datetime=[date(1980, 5, 1)])
        record = next(OMOPLoader().load(str(tmp_path)))
        assert record.demographics.birth_date == date(1980, 5, 1)

    def test_birth_date_from_year_month_day_columns(self, tmp_path: Path) -> None:
        _write_person(
            tmp_path, year_of_birth=[1980], month_of_birth=[5], day_of_birth=[1]
        )
        record = next(OMOPLoader().load(str(tmp_path)))
        assert record.demographics.birth_date == date(1980, 5, 1)

    def test_birth_date_defaults_month_and_day_when_absent(
        self, tmp_path: Path
    ) -> None:
        _write_person(tmp_path, year_of_birth=[1980])
        record = next(OMOPLoader().load(str(tmp_path)))
        assert record.demographics.birth_date == date(1980, 1, 1)

    def test_birth_date_degrades_on_an_invalid_day_of_month(
        self, tmp_path: Path
    ) -> None:
        _write_person(
            tmp_path, year_of_birth=[1980], month_of_birth=[2], day_of_birth=[31]
        )
        record = next(OMOPLoader().load(str(tmp_path)))
        assert record.demographics.birth_date == date(1980, 2, 1)

    def test_no_year_of_birth_leaves_birth_date_none(self, tmp_path: Path) -> None:
        _write_person(tmp_path)
        record = next(OMOPLoader().load(str(tmp_path)))
        assert record.demographics.birth_date is None

    def test_race_and_ethnicity_source_values(self, tmp_path: Path) -> None:
        _write_person(
            tmp_path,
            race_source_value=["White"],
            ethnicity_source_value=["Not Hispanic"],
        )
        record = next(OMOPLoader().load(str(tmp_path)))
        assert record.demographics.race == "White"
        assert record.demographics.ethnicity == "Not Hispanic"

    def test_deceased_is_always_false(self, tmp_path: Path) -> None:
        """See omop.py's own module docstring: no `death` table is read."""
        _write_person(tmp_path)
        record = next(OMOPLoader().load(str(tmp_path)))
        assert record.demographics.deceased is False


class TestVisitOccurrence:
    def test_visit_becomes_an_encounter(self, tmp_path: Path) -> None:
        _write_person(tmp_path)
        _write_table(
            tmp_path,
            "visit_occurrence.parquet",
            {
                "visit_occurrence_id": [100],
                "person_id": [1],
                "visit_source_value": ["inpatient"],
                "visit_start_date": [date(2024, 3, 14)],
                "visit_end_date": [date(2024, 3, 15)],
            },
        )
        record = next(OMOPLoader().load(str(tmp_path)))
        assert len(record.encounters) == 1
        encounter = record.encounters[0]
        assert encounter.encounter_id == "100"
        assert encounter.encounter_type == "inpatient"
        assert encounter.start == datetime(2024, 3, 14, tzinfo=UTC)
        assert encounter.end == datetime(2024, 3, 15, tzinfo=UTC)

    def test_prefers_datetime_columns_over_date_columns(self, tmp_path: Path) -> None:
        _write_person(tmp_path)
        _write_table(
            tmp_path,
            "visit_occurrence.parquet",
            {
                "visit_occurrence_id": [100],
                "person_id": [1],
                "visit_start_datetime": [
                    datetime(2024, 3, 14, 8, 30)  # noqa: DTZ001 -- real OMOP columns are naive
                ],
                "visit_start_date": [date(2024, 3, 14)],
                "visit_end_datetime": [None],
                "visit_end_date": [None],
            },
        )
        record = next(OMOPLoader().load(str(tmp_path)))
        assert record.encounters[0].start == datetime(2024, 3, 14, 8, 30, tzinfo=UTC)
        assert record.encounters[0].end is None

    def test_a_visit_for_an_unrelated_person_is_not_attached(
        self, tmp_path: Path
    ) -> None:
        _write_person(tmp_path, person_id=[1, 2])
        _write_table(
            tmp_path,
            "visit_occurrence.parquet",
            {"visit_occurrence_id": [100], "person_id": [2]},
        )
        records = {r.patient_id: r for r in OMOPLoader().load(str(tmp_path))}
        assert records["1"].encounters == []
        assert len(records["2"].encounters) == 1


class TestConditionOccurrence:
    def test_condition_source_value_becomes_a_coded_event(self, tmp_path: Path) -> None:
        _write_person(tmp_path)
        _write_table(
            tmp_path,
            "condition_occurrence.parquet",
            {
                "condition_occurrence_id": [1000],
                "person_id": [1],
                "condition_source_value": ["385093006"],
                "condition_start_date": [date(2024, 3, 14)],
                "visit_occurrence_id": [100],
                "condition_status_source_value": ["active"],
            },
        )
        record = next(OMOPLoader().load(str(tmp_path)))
        assert len(record.conditions) == 1
        event = record.conditions[0]
        assert event.code == "385093006"
        assert event.system == CodeSystem.SNOMED
        assert event.encounter_ref == "100"
        assert event.status == "active"

    def test_a_missing_condition_table_yields_no_conditions(
        self, tmp_path: Path
    ) -> None:
        _write_person(tmp_path)
        record = next(OMOPLoader().load(str(tmp_path)))
        assert record.conditions == []

    def test_a_row_with_no_source_value_is_dropped(self, tmp_path: Path) -> None:
        _write_person(tmp_path)
        _write_table(
            tmp_path,
            "condition_occurrence.parquet",
            {
                "condition_occurrence_id": [1000],
                "person_id": [1],
                "condition_source_value": [None],
            },
        )
        record = next(OMOPLoader().load(str(tmp_path)))
        assert record.conditions == []

    def test_configurable_condition_system(self, tmp_path: Path) -> None:
        _write_person(tmp_path)
        _write_table(
            tmp_path,
            "condition_occurrence.parquet",
            {
                "condition_occurrence_id": [1000],
                "person_id": [1],
                "condition_source_value": ["E11.9"],
            },
        )
        loader = OMOPLoader(condition_system=CodeSystem.ICD10CM)
        record = next(loader.load(str(tmp_path)))
        assert record.conditions[0].system == CodeSystem.ICD10CM


class TestDrugExposure:
    def test_drug_source_value_becomes_a_coded_event(self, tmp_path: Path) -> None:
        _write_person(tmp_path)
        _write_table(
            tmp_path,
            "drug_exposure.parquet",
            {
                "drug_exposure_id": [1],
                "person_id": [1],
                "drug_source_value": ["309090"],
                "drug_exposure_start_datetime": [
                    datetime(2024, 3, 14, 9, 0)  # noqa: DTZ001 -- real OMOP columns are naive
                ],
                "visit_occurrence_id": [100],
            },
        )
        record = next(OMOPLoader().load(str(tmp_path)))
        assert record.medications[0].code == "309090"
        assert record.medications[0].system == CodeSystem.RXNORM
        assert record.medications[0].timestamp == datetime(
            2024, 3, 14, 9, 0, tzinfo=UTC
        )

    def test_a_row_with_no_source_value_is_dropped(self, tmp_path: Path) -> None:
        _write_person(tmp_path)
        _write_table(
            tmp_path,
            "drug_exposure.parquet",
            {"drug_exposure_id": [1], "person_id": [1], "drug_source_value": [None]},
        )
        record = next(OMOPLoader().load(str(tmp_path)))
        assert record.medications == []


class TestProcedureOccurrence:
    def test_procedure_source_value_becomes_a_coded_event(self, tmp_path: Path) -> None:
        _write_person(tmp_path)
        _write_table(
            tmp_path,
            "procedure_occurrence.parquet",
            {
                "procedure_occurrence_id": [1],
                "person_id": [1],
                "procedure_source_value": ["71046"],
                "procedure_date": [date(2024, 3, 14)],
            },
        )
        record = next(OMOPLoader().load(str(tmp_path)))
        assert record.procedures[0].code == "71046"
        assert record.procedures[0].system == CodeSystem.CPT

    def test_a_row_with_no_source_value_is_dropped(self, tmp_path: Path) -> None:
        _write_person(tmp_path)
        _write_table(
            tmp_path,
            "procedure_occurrence.parquet",
            {
                "procedure_occurrence_id": [1],
                "person_id": [1],
                "procedure_source_value": [None],
            },
        )
        record = next(OMOPLoader().load(str(tmp_path)))
        assert record.procedures == []


class TestMeasurement:
    def test_measurement_becomes_an_observation(self, tmp_path: Path) -> None:
        _write_person(tmp_path)
        _write_table(
            tmp_path,
            "measurement.parquet",
            {
                "measurement_id": [1],
                "person_id": [1],
                "measurement_source_value": ["6690-2"],
                "value_as_number": [14.2],
                "unit_source_value": ["10*3/uL"],
                "range_low": [4.5],
                "range_high": [11.0],
                "measurement_datetime": [
                    datetime(2024, 3, 14, 8, 20)  # noqa: DTZ001 -- real OMOP columns are naive
                ],
                "visit_occurrence_id": [100],
            },
        )
        record = next(OMOPLoader().load(str(tmp_path)))
        obs = record.observations[0]
        assert obs.code == "6690-2"
        assert obs.system == CodeSystem.LOINC
        assert obs.value == 14.2
        assert obs.unit == "10*3/uL"
        assert obs.reference_range == (4.5, 11.0)
        assert obs.encounter_ref == "100"

    def test_falls_back_to_value_source_value_when_no_number(
        self, tmp_path: Path
    ) -> None:
        _write_person(tmp_path)
        _write_table(
            tmp_path,
            "measurement.parquet",
            {
                "measurement_id": [1],
                "person_id": [1],
                "measurement_source_value": ["X"],
                "value_source_value": ["Positive"],
            },
        )
        record = next(OMOPLoader().load(str(tmp_path)))
        assert record.observations[0].value == "Positive"

    def test_missing_range_bound_leaves_reference_range_none(
        self, tmp_path: Path
    ) -> None:
        _write_person(tmp_path)
        _write_table(
            tmp_path,
            "measurement.parquet",
            {
                "measurement_id": [1],
                "person_id": [1],
                "measurement_source_value": ["6690-2"],
                "range_low": [None],
                "range_high": [11.0],
            },
        )
        record = next(OMOPLoader().load(str(tmp_path)))
        assert record.observations[0].reference_range is None

    def test_a_row_with_no_source_value_is_dropped(self, tmp_path: Path) -> None:
        _write_person(tmp_path)
        _write_table(
            tmp_path,
            "measurement.parquet",
            {
                "measurement_id": [1],
                "person_id": [1],
                "measurement_source_value": [None],
            },
        )
        record = next(OMOPLoader().load(str(tmp_path)))
        assert record.observations == []


class TestRegistration:
    def test_registered_under_the_expected_key(self) -> None:
        assert OMOPLoader.registry_key == "loader.ehr.omop"

    def test_provenance_is_serialisable(self) -> None:
        prov = OMOPLoader().provenance()
        assert isinstance(prov.model_dump_json(), str)
