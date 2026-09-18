"""``PatientTimelineSerializer``: the cross-modal seam (docs/05_DATA_MODALITY_SPEC.md
section 2.3) -- converts a ``PatientRecord`` (``openbtk.data.ehr``) into a
``ClinicalTextRecord`` (``openbtk.data.clinical_text``), so a structured
chart flows into the SAME text pipeline (chunking, embedding, retrieval,
de-identification, guardrails) clinical notes already use, with no new
machinery.

**Lives in ``openbtk.pipelines``, not ``openbtk.data.ehr``.** A converter
between two modality schemas genuinely needs to import both concrete types
-- but import-linter's own "Modalities are independent of one another"
contract (pyproject.toml) forbids ``openbtk.data.ehr`` from importing
``openbtk.data.clinical_text`` (verified directly: placing this class under
``openbtk.data.ehr`` and running ``pre-commit``'s import-linter hook fails
with exactly that violation, not a hypothetical one). ``openbtk.pipelines``
sits one layer above ``openbtk.data`` in the project's own layered-
architecture contract and may depend on either or both modalities --
:class:`~openbtk.pipelines.rag.RAGPipeline` already does the analogous
thing across provider categories. This is the correct home, not a
workaround.

**Not a registered pipeline step**, for a separate reason:
``BasePreprocessor.process()`` is deliberately same-type in and out
(``RecordT -> RecordT``, core/base.py) -- a pipeline step never silently
changes a record's type mid-chain, which is the entire reason that class
has one type parameter, not two. A converter whose entire purpose is to
turn a ``PatientRecord`` into a ``ClinicalTextRecord`` genuinely does not
fit that contract, and forcing it in would be exactly the "edit core to
make one modality fit" CLAUDE.md rule 8 exists to stop.
``PatientTimelineSerializer`` is instead a plain class with a
``serialize()`` method, the same treatment docs/05_DATA_MODALITY_SPEC.md's
own component table gives ``CohortBuilder`` (task 6.6) for an analogous
reason ("-- (not a pipeline step)").

**Rendering is chronological, compact, and includes codes inline** (exactly
as section 2.3 specifies) so retrieval can match on either the display text
or the code. ``deid_status`` propagates unchanged from the source
``PatientRecord`` -- both modalities share the exact same
``openbtk.deid.schemas.DeidStatus`` enum, which is what makes this
propagation a plain field copy rather than a conversion.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from openbtk.data.clinical_text.schemas import ClinicalTextRecord

if TYPE_CHECKING:
    from collections.abc import Callable

    from openbtk.data.ehr.schemas import (
        CodedEvent,
        Encounter,
        Measurement,
        PatientRecord,
    )

_EPOCH = datetime.min.replace(tzinfo=UTC)

# Natural-language noun for a rendered encounter line ("Inpatient
# admission", matching docs/05_DATA_MODALITY_SPEC.md section 2.3's own
# worked example) -- a documented default, not a universal claim about
# every possible encounter_type string a source system might use.
_ENCOUNTER_TYPE_NOUN: dict[str, str] = {
    "inpatient": "admission",
    "ambulatory": "visit",
    "outpatient": "visit",
    "emergency": "visit",
}


def _default_encounter_template(encounter: Encounter) -> str:
    kind = encounter.encounter_type
    noun = _ENCOUNTER_TYPE_NOUN.get((kind or "").lower(), "encounter")
    label = kind.capitalize() if kind else "Unspecified"
    return f"{label} {noun}"


def _default_coded_event_template(event: CodedEvent) -> str:
    name = event.display or event.code
    return f"{name} ({event.system.value} {event.code})"


def _range_flag(value: float | str | None, low: float, high: float) -> str | None:
    if not isinstance(value, (int, float)):
        return None
    if value > high:
        return "HIGH"
    if value < low:
        return "LOW"
    return None


def _default_observation_template(observation: Measurement) -> str:
    name = observation.display or observation.code
    parts = [name]
    if observation.value is not None:
        value_str = str(observation.value)
        if observation.unit:
            value_str += f" {observation.unit}"
        parts.append(value_str)
    if observation.reference_range is not None:
        low, high = observation.reference_range
        parts.append(f"(ref {low}-{high})")
        flag = _range_flag(observation.value, low, high)
        if flag is not None:
            parts.append(f"[{flag}]")
    parts.append(f"({observation.system.value} {observation.code})")
    return " ".join(parts)


class PatientTimelineSerializer:
    """Render a ``PatientRecord`` as a chronological plain-text timeline.

    Args:
        encounter_label: Column label for encounter rows.
        condition_label: Column label for condition rows.
        medication_label: Column label for medication rows.
        procedure_label: Column label for procedure rows.
        observation_label: Column label for observation/lab rows.
        encounter_template: Overrides the default one-line encounter
            description. Same idea for the other four ``*_template`` args.
        source: ``ClinicalTextRecord.source`` value for every record this
            serializer produces.

    Example:
        >>> from datetime import datetime, UTC
        >>> from openbtk.core.schemas import CodeSystem
        >>> from openbtk.data.ehr.schemas import CodedEvent, Demographics, PatientRecord
        >>> record = PatientRecord(
        ...     patient_id="pt-1",
        ...     demographics=Demographics(),
        ...     conditions=[CodedEvent(
        ...         code="385093006", system=CodeSystem.SNOMED,
        ...         display="Community-acquired pneumonia",
        ...         timestamp=datetime(2024, 3, 14, tzinfo=UTC),
        ...     )],
        ...     source_system="fhir-r4",
        ... )
        >>> text_record = PatientTimelineSerializer().serialize(record)
        >>> "Community-acquired pneumonia (SNOMED 385093006)" in text_record.text
        True
        >>> text_record.patient_ref
        'pt-1'
    """

    def __init__(
        self,
        *,
        encounter_label: str = "Encounter",
        condition_label: str = "Condition",
        medication_label: str = "Medication",
        procedure_label: str = "Procedure",
        observation_label: str = "Lab",
        encounter_template: Callable[[Encounter], str] = _default_encounter_template,
        condition_template: Callable[[CodedEvent], str] = _default_coded_event_template,
        medication_template: Callable[
            [CodedEvent], str
        ] = _default_coded_event_template,
        procedure_template: Callable[[CodedEvent], str] = _default_coded_event_template,
        observation_template: Callable[
            [Measurement], str
        ] = _default_observation_template,
        source: str = "ehr-timeline",
    ) -> None:
        self._encounter_label = encounter_label
        self._condition_label = condition_label
        self._medication_label = medication_label
        self._procedure_label = procedure_label
        self._observation_label = observation_label
        self._encounter_template = encounter_template
        self._condition_template = condition_template
        self._medication_template = medication_template
        self._procedure_template = procedure_template
        self._observation_template = observation_template
        self._source = source

    def serialize(self, patient: PatientRecord) -> ClinicalTextRecord:
        """Render every encounter/condition/medication/procedure/
        observation as one chronological, compact plain-text timeline.

        Returns:
            A ``ClinicalTextRecord`` whose ``patient_ref`` and
            ``deid_status`` carry over from ``patient`` unchanged, ready
            for the same de-identification/chunking/guardrail pipeline
            any other clinical text record goes through.
        """
        rows: list[tuple[datetime | None, str, str]] = []
        for encounter in patient.encounters:
            rows.append(
                (
                    encounter.start,
                    self._encounter_label,
                    self._encounter_template(encounter),
                )
            )
        for condition in patient.conditions:
            rows.append(
                (
                    condition.timestamp,
                    self._condition_label,
                    self._condition_template(condition),
                )
            )
        for medication in patient.medications:
            rows.append(
                (
                    medication.timestamp,
                    self._medication_label,
                    self._medication_template(medication),
                )
            )
        for procedure in patient.procedures:
            rows.append(
                (
                    procedure.timestamp,
                    self._procedure_label,
                    self._procedure_template(procedure),
                )
            )
        for observation in patient.observations:
            rows.append(
                (
                    observation.timestamp,
                    self._observation_label,
                    self._observation_template(observation),
                )
            )
        rows.sort(key=lambda row: (row[0] is None, row[0] or _EPOCH))
        text = "\n".join(self._render_line(*row) for row in rows)
        return ClinicalTextRecord(
            record_id=f"timeline:{patient.patient_id}",
            source=self._source,
            text=text,
            note_type="EHR Timeline",
            patient_ref=patient.patient_id,
            deid_status=patient.deid_status,
        )

    @staticmethod
    def _render_line(timestamp: datetime | None, label: str, description: str) -> str:
        date_str = timestamp.date().isoformat() if timestamp is not None else "undated"
        return f"{date_str:<10} | {label:<10} | {description}"
