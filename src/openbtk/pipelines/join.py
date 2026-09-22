"""Cross-modal joins: attach a patient's structured events to their notes (FR-E-08).

Every clinical note carries an optional ``patient_ref`` and ``encounter_ref`` and every
``PatientRecord`` has a ``patient_id`` (docs/05_DATA_MODALITY_SPEC.md section 3), so a
note can be joined to the codes, medications, procedures and measurements recorded for
the same patient: *"the discharge summary, plus the labs drawn in the day before it"*.

``join_notes_to_events`` does this as a stream over notes. An event joins a note when:

* ``on="encounter"``: it shares the note's ``encounter_ref``;
* ``on="window"``: its timestamp is within ``before`` of the note's timestamp, or within
  ``after`` past it (a note is normally written after the events it describes, so the
  default looks back 24 hours and not forward);
* ``on="either"`` (the default): either of the above. An event that matches both is
  reported as an encounter match.

**Lives in ``openbtk.pipelines``**, not in either modality: it needs both concrete
types, and import-linter forbids the two modality packages from importing each other
(see ``openbtk.pipelines.timeline`` for the same reasoning).

What it does not do:

* It joins on identifiers, exactly. It does not link records by name, date of birth or
  any other attribute, and it will not guess: a note whose ``patient_ref`` is missing or
  unknown comes back with ``patient_found=False`` and no events.
* ``patient_ref`` must equal ``PatientRecord.patient_id``. If your two sources
  pseudonymise differently, map one to the other first; a join over mismatched
  identifiers matches nothing, silently, which is why ``patient_found`` exists to count.
* Memory: notes stream, one at a time. Patients are looked up, so they are held as a
  mapping (``index_patients`` builds one, at one entry per *patient*) or fetched on
  demand by a function you supply.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from openbtk.data.clinical_text.schemas import ClinicalTextRecord  # noqa: TC001
from openbtk.data.ehr.schemas import CodedEvent, Measurement  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from openbtk.data.ehr.schemas import PatientRecord

JoinOn = Literal["encounter", "window", "either"]

_UNDATED = datetime.min.replace(tzinfo=UTC)

_KINDS = (
    ("conditions", "condition"),
    ("medications", "medication"),
    ("procedures", "procedure"),
    ("observations", "observation"),
)


class MatchedEvent(BaseModel):
    """One structured event joined to a note, and why it matched.

    Example:
        >>> from openbtk.core.schemas import CodeSystem
        >>> matched = MatchedEvent(
        ...     kind="condition",
        ...     event=CodedEvent(code="385093006", system=CodeSystem.SNOMED),
        ...     matched_by="encounter",
        ... )
        >>> matched.kind
        'condition'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["condition", "medication", "procedure", "observation"]
    event: CodedEvent | Measurement
    matched_by: Literal["encounter", "window"]


class NoteWithEvents(BaseModel):
    """A note and the structured events joined to it, in time order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    note: ClinicalTextRecord
    patient_found: bool = Field(
        ...,
        description="False when the note has no patient_ref or it matched no patient.",
    )
    events: list[MatchedEvent] = Field(default_factory=list)


def index_patients(patients: Iterable[PatientRecord]) -> dict[str, PatientRecord]:
    """A lookup of ``patients`` by ``patient_id`` for ``join_notes_to_events``.

    Holds every patient record, so for a large cohort supply a function that fetches a
    patient on demand instead. A repeated ``patient_id`` keeps the last record.

    Example:
        >>> from openbtk.data.ehr.schemas import Demographics, PatientRecord
        >>> record = PatientRecord(
        ...     patient_id="pt-1", demographics=Demographics(), source_system="fhir-r4"
        ... )
        >>> list(index_patients([record]))
        ['pt-1']
    """
    return {patient.patient_id: patient for patient in patients}


def _matched_by(
    note: ClinicalTextRecord,
    event: CodedEvent | Measurement,
    on: JoinOn,
    before: timedelta,
    after: timedelta,
) -> Literal["encounter", "window"] | None:
    if (
        on in ("encounter", "either")
        and note.encounter_ref is not None
        and event.encounter_ref == note.encounter_ref
    ):
        return "encounter"
    if (
        on in ("window", "either")
        and note.timestamp is not None
        and event.timestamp is not None
        and note.timestamp - before <= event.timestamp <= note.timestamp + after
    ):
        return "window"
    return None


def join_notes_to_events(
    notes: Iterable[ClinicalTextRecord],
    patients: Mapping[str, PatientRecord] | Callable[[str], PatientRecord | None],
    *,
    on: JoinOn = "either",
    before: timedelta = timedelta(hours=24),
    after: timedelta = timedelta(0),
) -> Iterator[NoteWithEvents]:
    """Join each note to its patient's events. Yields one result per note, in order.

    Args:
        notes: The notes, streamed.
        patients: A mapping of ``patient_id`` to record (see :func:`index_patients`), or
            a function returning the record for an id (or ``None``).
        on: ``"encounter"``, ``"window"`` or ``"either"``; see the module docstring.
        before: How far before a note's timestamp an event may be and still join.
        after: How far after it.

    Raises:
        ValueError: If ``before`` or ``after`` is negative.

    Example:
        >>> from datetime import UTC, datetime
        >>> from openbtk.core.schemas import CodeSystem
        >>> from openbtk.data.ehr.schemas import (
        ...     Demographics, Measurement, PatientRecord,
        ... )
        >>> noon = datetime(2024, 3, 14, 12, tzinfo=UTC)
        >>> patient = PatientRecord(
        ...     patient_id="pt-1",
        ...     demographics=Demographics(),
        ...     observations=[Measurement(code="718-7", value=13.5, timestamp=noon)],
        ...     source_system="fhir-r4",
        ... )
        >>> note = ClinicalTextRecord(
        ...     record_id="n1", source="s", text="Discharge summary.",
        ...     patient_ref="pt-1", timestamp=noon.replace(hour=18),
        ... )
        >>> (joined,) = join_notes_to_events([note], index_patients([patient]))
        >>> matches = [(e.kind, e.matched_by) for e in joined.events]
        >>> matches
        [('observation', 'window')]
    """
    if before < timedelta(0) or after < timedelta(0):
        raise ValueError("before and after must not be negative")
    return _join(notes, patients, on, before, after)


def _join(
    notes: Iterable[ClinicalTextRecord],
    patients: Mapping[str, PatientRecord] | Callable[[str], PatientRecord | None],
    on: JoinOn,
    before: timedelta,
    after: timedelta,
) -> Iterator[NoteWithEvents]:
    def find(ref: str) -> PatientRecord | None:
        return patients.get(ref) if isinstance(patients, Mapping) else patients(ref)

    for note in notes:
        patient = find(note.patient_ref) if note.patient_ref is not None else None
        if patient is None:
            yield NoteWithEvents(note=note, patient_found=False)
            continue
        matched: list[MatchedEvent] = []
        for attribute, kind in _KINDS:
            for event in getattr(patient, attribute):
                how = _matched_by(note, event, on, before, after)
                if how is not None:
                    matched.append(MatchedEvent(kind=kind, event=event, matched_by=how))
        matched.sort(
            key=lambda m: (m.event.timestamp is None, m.event.timestamp or _UNDATED)
        )
        yield NoteWithEvents(note=note, patient_found=True, events=matched)
