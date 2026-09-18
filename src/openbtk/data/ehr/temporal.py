"""``TemporalNormalizer``: encounter-anchored timeline normalisation for a
``PatientRecord`` (docs/05_DATA_MODALITY_SPEC.md section 2.2).

Two real data-quality problems this addresses, both common in real EHR
exports:

1. **Missing event timestamps.** A condition/medication/procedure/
   observation sometimes carries a known encounter but no event-level
   timestamp of its own (the source system only timestamped the visit, not
   every entry within it -- OMOP's ``visit_occurrence`` table in particular
   frequently has this shape). When ``anchor_missing_timestamps`` is True
   (the default), such an event borrows its encounter's ``start`` time --
   a disclosed approximation, not a fabricated fact -- and stays ``None``
   when its ``encounter_ref`` does not resolve to a known encounter, or
   that encounter itself has no ``start``.
2. **Out-of-order events.** Loaders yield events in whatever order their
   source format happens to store them (OMOP's Parquet row order, a FHIR
   bundle's entry order) -- not necessarily chronological. ``process()``
   returns every list (encounters, conditions, medications, procedures,
   observations) sorted by timestamp, undated items last and otherwise
   stable, so ``PatientTimelineSerializer`` (task 6.5) can render a real
   chronology without re-implementing this itself.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TypeVar

from openbtk.core.base import BasePreprocessor
from openbtk.core.registry import PREPROCESSOR_REGISTRY
from openbtk.data.ehr.schemas import CodedEvent, Encounter, Measurement, PatientRecord

_EventT = TypeVar("_EventT", CodedEvent, Measurement)

_EPOCH = datetime.min.replace(tzinfo=UTC)
"""Sort-key filler for a ``None`` timestamp -- paired with the ``is None``
flag below so undated items still sort deterministically last, never
crashing on a ``None``-vs-``datetime`` comparison."""


def _encounter_sort_key(encounter: Encounter) -> tuple[bool, datetime]:
    return (encounter.start is None, encounter.start or _EPOCH)


def _event_sort_key(event: CodedEvent | Measurement) -> tuple[bool, datetime]:
    return (event.timestamp is None, event.timestamp or _EPOCH)


@PREPROCESSOR_REGISTRY.register("preprocessor.ehr.temporal")
class TemporalNormalizer(BasePreprocessor[PatientRecord]):
    """Anchor missing event timestamps to their encounter and sort every
    event list chronologically.

    Args:
        anchor_missing_timestamps: When True (default), an event with no
            timestamp of its own but a resolvable ``encounter_ref``
            borrows that encounter's ``start`` time.

    Example:
        >>> from datetime import datetime, UTC
        >>> from openbtk.core.schemas import CodeSystem
        >>> from openbtk.data.ehr.schemas import (
        ...     CodedEvent, Demographics, Encounter, PatientRecord,
        ... )
        >>> record = PatientRecord(
        ...     patient_id="pt-1",
        ...     demographics=Demographics(),
        ...     encounters=[Encounter(
        ...         encounter_id="enc-1",
        ...         start=datetime(2024, 3, 14, 8, tzinfo=UTC),
        ...     )],
        ...     conditions=[CodedEvent(
        ...         code="385093006", system=CodeSystem.SNOMED, encounter_ref="enc-1",
        ...     )],
        ...     source_system="fhir-r4",
        ... )
        >>> normalized = TemporalNormalizer().process(record)
        >>> normalized.conditions[0].timestamp
        datetime.datetime(2024, 3, 14, 8, 0, tzinfo=datetime.timezone.utc)
    """

    def __init__(self, *, anchor_missing_timestamps: bool = True) -> None:
        self._anchor_missing_timestamps = anchor_missing_timestamps

    def process(self, record: PatientRecord) -> PatientRecord:
        encounters_by_id = {e.encounter_id: e for e in record.encounters}
        return record.model_copy(
            update={
                "encounters": sorted(record.encounters, key=_encounter_sort_key),
                "conditions": self._normalize_events(
                    record.conditions, encounters_by_id
                ),
                "medications": self._normalize_events(
                    record.medications, encounters_by_id
                ),
                "procedures": self._normalize_events(
                    record.procedures, encounters_by_id
                ),
                "observations": self._normalize_events(
                    record.observations, encounters_by_id
                ),
            }
        )

    def _normalize_events(
        self, events: list[_EventT], encounters_by_id: dict[str, Encounter]
    ) -> list[_EventT]:
        anchored = [self._anchor(event, encounters_by_id) for event in events]
        return sorted(anchored, key=_event_sort_key)

    def _anchor(
        self, event: _EventT, encounters_by_id: dict[str, Encounter]
    ) -> _EventT:
        if not self._anchor_missing_timestamps or event.timestamp is not None:
            return event
        if event.encounter_ref is None:
            return event
        encounter = encounters_by_id.get(event.encounter_ref)
        if encounter is None or encounter.start is None:
            return event
        return event.model_copy(update={"timestamp": encounter.start})
