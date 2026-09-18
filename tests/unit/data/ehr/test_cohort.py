"""Unit tests for openbtk.data.ehr.cohort."""

from __future__ import annotations

from datetime import date

from openbtk.core.schemas import CodeSystem
from openbtk.data.ehr.cohort import (
    CohortBuilder,
    age_between,
    has_condition,
    has_medication,
    has_procedure,
)
from openbtk.data.ehr.schemas import CodedEvent, Demographics, PatientRecord


def _patient(**overrides: object) -> PatientRecord:
    defaults: dict[str, object] = {
        "patient_id": "pt-1",
        "demographics": Demographics(),
        "source_system": "fhir-r4",
    }
    defaults.update(overrides)
    return PatientRecord(**defaults)  # type: ignore[arg-type]


class TestHasConditionMedicationProcedure:
    def test_has_condition_matches_by_code(self) -> None:
        patient = _patient(
            conditions=[CodedEvent(code="385093006", system=CodeSystem.SNOMED)]
        )
        assert has_condition("385093006")(patient) is True
        assert has_condition("other")(patient) is False

    def test_has_condition_narrows_by_system(self) -> None:
        patient = _patient(conditions=[CodedEvent(code="X", system=CodeSystem.SNOMED)])
        assert has_condition("X", system=CodeSystem.SNOMED)(patient) is True
        assert has_condition("X", system=CodeSystem.ICD10CM)(patient) is False

    def test_has_medication_matches_by_code(self) -> None:
        patient = _patient(
            medications=[CodedEvent(code="309090", system=CodeSystem.RXNORM)]
        )
        assert has_medication("309090")(patient) is True
        assert has_medication("other")(patient) is False

    def test_has_procedure_matches_by_code(self) -> None:
        patient = _patient(procedures=[CodedEvent(code="71046", system=CodeSystem.CPT)])
        assert has_procedure("71046")(patient) is True
        assert has_procedure("other")(patient) is False

    def test_no_matching_events_is_false(self) -> None:
        patient = _patient()
        assert has_condition("x")(patient) is False


class TestAgeBetween:
    def test_matches_within_range(self) -> None:
        patient = _patient(demographics=Demographics(birth_date=date(1990, 1, 1)))
        predicate = age_between(18, 89, as_of=date(2024, 1, 1))
        assert predicate(patient) is True

    def test_excludes_below_range(self) -> None:
        patient = _patient(demographics=Demographics(birth_date=date(2020, 1, 1)))
        predicate = age_between(18, 89, as_of=date(2024, 1, 1))
        assert predicate(patient) is False

    def test_excludes_above_range(self) -> None:
        patient = _patient(demographics=Demographics(birth_date=date(1900, 1, 1)))
        predicate = age_between(18, 89, as_of=date(2024, 1, 1))
        assert predicate(patient) is False

    def test_birthday_not_yet_reached_this_year(self) -> None:
        # Born 1990-06-15; as_of 2024-06-01 -> still 33, not yet 34.
        patient = _patient(demographics=Demographics(birth_date=date(1990, 6, 15)))
        predicate = age_between(34, 34, as_of=date(2024, 6, 1))
        assert predicate(patient) is False
        predicate_33 = age_between(33, 33, as_of=date(2024, 6, 1))
        assert predicate_33(patient) is True

    def test_birthday_already_passed_this_year(self) -> None:
        patient = _patient(demographics=Demographics(birth_date=date(1990, 6, 15)))
        predicate = age_between(34, 34, as_of=date(2024, 6, 20))
        assert predicate(patient) is True

    def test_unknown_birth_date_never_matches(self) -> None:
        patient = _patient(demographics=Demographics())
        predicate = age_between(0, 120, as_of=date(2024, 1, 1))
        assert predicate(patient) is False

    def test_defaults_as_of_to_today(self) -> None:
        patient = _patient(demographics=Demographics(birth_date=date(2000, 1, 1)))
        predicate = age_between(0, 200)
        assert predicate(patient) is True


class TestCohortBuilder:
    def test_include_filters_to_matching_patients(self) -> None:
        matching = _patient(
            patient_id="pt-1",
            conditions=[CodedEvent(code="X", system=CodeSystem.SNOMED)],
        )
        other = _patient(patient_id="pt-2")
        cohort = CohortBuilder([matching, other]).include(has_condition("X"))
        assert [p.patient_id for p in cohort] == ["pt-1"]

    def test_multiple_includes_are_logical_and(self) -> None:
        patient = _patient(
            patient_id="pt-1",
            conditions=[CodedEvent(code="X", system=CodeSystem.SNOMED)],
        )
        cohort = (
            CohortBuilder([patient])
            .include(has_condition("X"))
            .include(has_condition("Y"))
        )
        assert list(cohort) == []

    def test_exclude_drops_matching_patients(self) -> None:
        patient = _patient(
            patient_id="pt-1",
            medications=[CodedEvent(code="M", system=CodeSystem.RXNORM)],
        )
        cohort = CohortBuilder([patient]).exclude(has_medication("M"))
        assert list(cohort) == []

    def test_multiple_excludes_are_logical_or(self) -> None:
        patient = _patient(
            patient_id="pt-1",
            medications=[CodedEvent(code="M", system=CodeSystem.RXNORM)],
        )
        cohort = (
            CohortBuilder([patient])
            .exclude(has_medication("other"))
            .exclude(has_medication("M"))
        )
        assert list(cohort) == []

    def test_no_predicates_yields_everything(self) -> None:
        patients = [_patient(patient_id="pt-1"), _patient(patient_id="pt-2")]
        cohort = CohortBuilder(patients)
        assert [p.patient_id for p in cohort] == ["pt-1", "pt-2"]

    def test_include_and_exclude_together(self) -> None:
        patient = _patient(
            patient_id="pt-1",
            conditions=[CodedEvent(code="X", system=CodeSystem.SNOMED)],
            medications=[CodedEvent(code="M", system=CodeSystem.RXNORM)],
        )
        cohort = (
            CohortBuilder([patient])
            .include(has_condition("X"))
            .exclude(has_medication("M"))
        )
        assert list(cohort) == []

    def test_streams_never_materialises(self) -> None:
        """A generator source is consumed lazily -- CohortBuilder never
        calls list()/tuple() on it internally (ADR-0004)."""

        def _source() -> object:
            yield _patient(patient_id="pt-1")
            raise AssertionError("must not be reached for top_k=1-style early exit")

        cohort = iter(CohortBuilder(_source()))
        first = next(cohort)
        assert first.patient_id == "pt-1"

    def test_include_returns_self_for_chaining(self) -> None:
        builder = CohortBuilder([])
        assert builder.include(has_condition("x")) is builder

    def test_exclude_returns_self_for_chaining(self) -> None:
        builder = CohortBuilder([])
        assert builder.exclude(has_condition("x")) is builder
