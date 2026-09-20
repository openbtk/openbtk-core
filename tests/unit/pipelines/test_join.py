"""Cross-modal note/event joins (FR-E-08). All records are synthetic."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from openbtk.core.schemas import CodeSystem
from openbtk.data.clinical_text.schemas import ClinicalTextRecord
from openbtk.data.ehr.schemas import (
    CodedEvent,
    Demographics,
    Measurement,
    PatientRecord,
)
from openbtk.pipelines.join import index_patients, join_notes_to_events

if TYPE_CHECKING:
    from collections.abc import Iterator

NOON = datetime(2024, 3, 14, 12, tzinfo=UTC)


def _hours(n: float) -> datetime:
    return NOON + timedelta(hours=n)


def _patient(**events: list[object]) -> PatientRecord:
    return PatientRecord(
        patient_id="pt-1",
        demographics=Demographics(),
        source_system="fhir-r4",
        **events,  # type: ignore[arg-type]
    )


def _note(
    hours: float | None = 0,
    *,
    patient_ref: str | None = "pt-1",
    encounter_ref: str | None = None,
    record_id: str = "n1",
) -> ClinicalTextRecord:
    return ClinicalTextRecord(
        record_id=record_id,
        source="synthetic",
        text="A note.",
        patient_ref=patient_ref,
        encounter_ref=encounter_ref,
        timestamp=None if hours is None else _hours(hours),
    )


def _obs(hours: float | None, code: str = "718-7", **kwargs: object) -> Measurement:
    return Measurement(
        code=code,
        value=1.0,
        timestamp=None if hours is None else _hours(hours),
        **kwargs,  # type: ignore[arg-type]
    )


def _one(note: ClinicalTextRecord, patient: PatientRecord, **kwargs: object):  # type: ignore[no-untyped-def]
    (joined,) = join_notes_to_events([note], index_patients([patient]), **kwargs)  # type: ignore[arg-type]
    return joined


class TestWindow:
    def test_an_event_inside_the_lookback_joins(self) -> None:
        joined = _one(_note(0), _patient(observations=[_obs(-3)]), on="window")
        assert [(e.kind, e.matched_by) for e in joined.events] == [
            ("observation", "window")
        ]

    def test_the_default_looks_back_a_day_and_not_forward(self) -> None:
        patient = _patient(
            observations=[_obs(-24, "a"), _obs(-24.01, "b"), _obs(0.01, "c")]
        )
        codes = [e.event.code for e in _one(_note(0), patient).events]
        assert codes == ["a"]

    def test_the_window_is_configurable_in_both_directions(self) -> None:
        patient = _patient(observations=[_obs(-2, "early"), _obs(1, "late")])
        joined = _one(
            _note(0),
            patient,
            on="window",
            before=timedelta(hours=3),
            after=timedelta(hours=2),
        )
        assert [e.event.code for e in joined.events] == ["early", "late"]

    def test_the_boundaries_are_inclusive(self) -> None:
        patient = _patient(observations=[_obs(-1, "lo"), _obs(1, "hi")])
        joined = _one(
            _note(0),
            patient,
            on="window",
            before=timedelta(hours=1),
            after=timedelta(hours=1),
        )
        assert len(joined.events) == 2

    def test_an_undated_note_or_event_never_matches_by_window(self) -> None:
        assert (
            _one(_note(None), _patient(observations=[_obs(0)]), on="window").events
            == []
        )
        assert (
            _one(_note(0), _patient(observations=[_obs(None)]), on="window").events
            == []
        )

    def test_negative_windows_are_refused_at_call_time(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            join_notes_to_events([], {}, before=timedelta(hours=-1))
        with pytest.raises(ValueError, match="negative"):
            join_notes_to_events([], {}, after=timedelta(hours=-1))


class TestEncounter:
    def test_events_of_the_same_encounter_join_whatever_their_time(self) -> None:
        patient = _patient(observations=[_obs(-500, encounter_ref="e1"), _obs(0, "x")])
        joined = _one(_note(0, encounter_ref="e1"), patient, on="encounter")
        assert [e.matched_by for e in joined.events] == ["encounter"]

    def test_a_note_without_an_encounter_matches_nothing_by_encounter(self) -> None:
        patient = _patient(observations=[_obs(0, encounter_ref="e1"), _obs(0)])
        assert _one(_note(0), patient, on="encounter").events == []

    def test_either_reports_an_encounter_match_as_such(self) -> None:
        patient = _patient(observations=[_obs(-1, encounter_ref="e1")])
        joined = _one(_note(0, encounter_ref="e1"), patient)
        assert [e.matched_by for e in joined.events] == ["encounter"]

    def test_either_takes_the_union(self) -> None:
        patient = _patient(
            observations=[_obs(-100, "enc", encounter_ref="e1"), _obs(-1, "win")]
        )
        joined = _one(_note(0, encounter_ref="e1"), patient)
        assert {e.event.code: e.matched_by for e in joined.events} == {
            "enc": "encounter",
            "win": "window",
        }


class TestEventKinds:
    def test_all_four_kinds_join_and_are_labelled(self) -> None:
        def coded(code: str) -> CodedEvent:
            return CodedEvent(code=code, system=CodeSystem.SNOMED, timestamp=_hours(-1))

        patient = _patient(
            conditions=[coded("c")],
            medications=[coded("m")],
            procedures=[coded("p")],
            observations=[_obs(-1, "o")],
        )
        joined = _one(_note(0), patient)
        assert {e.kind for e in joined.events} == {
            "condition",
            "medication",
            "procedure",
            "observation",
        }

    def test_events_come_back_in_time_order_undated_last(self) -> None:
        patient = _patient(
            observations=[_obs(-1, "late"), _obs(-5, "early")],
            conditions=[
                CodedEvent(code="undated", system=CodeSystem.SNOMED, encounter_ref="e1")
            ],
        )
        joined = _one(_note(0, encounter_ref="e1"), patient)
        assert [e.event.code for e in joined.events] == ["early", "late", "undated"]


class TestPatients:
    def test_an_unknown_patient_is_reported_not_guessed(self) -> None:
        joined = _one(
            _note(0, patient_ref="someone-else"), _patient(observations=[_obs(0)])
        )
        assert joined.patient_found is False and joined.events == []

    def test_a_note_without_a_patient_ref_is_reported(self) -> None:
        joined = _one(_note(0, patient_ref=None), _patient(observations=[_obs(0)]))
        assert joined.patient_found is False

    def test_a_found_patient_with_no_matching_events_is_still_found(self) -> None:
        joined = _one(_note(0), _patient())
        assert joined.patient_found is True and joined.events == []

    def test_a_lookup_function_may_stand_in_for_a_mapping(self) -> None:
        patient = _patient(observations=[_obs(-1)])
        calls: list[str] = []

        def fetch(ref: str) -> PatientRecord | None:
            calls.append(ref)
            return patient if ref == "pt-1" else None

        notes = [_note(0), _note(0, patient_ref="nope", record_id="n2")]
        joined = list(join_notes_to_events(notes, fetch))
        assert [j.patient_found for j in joined] == [True, False]
        assert calls == ["pt-1", "nope"]

    def test_index_patients_keys_by_patient_id_and_the_last_wins(self) -> None:
        first, second = _patient(), _patient(observations=[_obs(0)])
        assert index_patients([first, second])["pt-1"] is second


class TestStreaming:
    def test_notes_are_consumed_one_at_a_time(self) -> None:
        consumed = 0

        def notes() -> Iterator[ClinicalTextRecord]:
            nonlocal consumed
            for i in range(5):
                consumed += 1
                yield _note(0, record_id=f"n{i}")

        results = join_notes_to_events(notes(), {})
        next(results)
        assert consumed == 1

    def test_the_input_order_is_preserved(self) -> None:
        notes = [_note(0, record_id=f"n{i}") for i in range(4)]
        out = [j.note.record_id for j in join_notes_to_events(notes, {})]
        assert out == ["n0", "n1", "n2", "n3"]


def test_the_result_is_serialisable() -> None:
    joined = _one(_note(0), _patient(observations=[_obs(-1)]))
    assert '"observation"' in joined.model_dump_json()
