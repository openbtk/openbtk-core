"""EHR record schemas (docs/05_DATA_MODALITY_SPEC.md section 2.1).

``PatientRecord`` is the convergence point FHIR and OMOP both map into, so
downstream components (``TemporalNormalizer``, ``PatientTimelineSerializer``,
``CohortBuilder``) are source-agnostic -- "switch from FHIR to OMOP" becomes
a config change, not a rewrite.

``DeidStatus`` lives in ``openbtk.deid.schemas``, not here -- the same
reasoning as ``openbtk.data.clinical_text.schemas``: both modalities need it,
and the layering rule (docs/03_ARCHITECTURE.md section 2) keeps them
independent of each other, so a concept both need belongs in a module both
already depend on downward.

``CodeSystem`` (``openbtk.core.schemas``) resolves a real documentation
inconsistency that module's own docstring flagged as "to reconcile when the
EHR module is built (M6)": docs/05_DATA_MODALITY_SPEC.md's original
``CodedEvent.system`` comment listed a different set (SNOMED, ICD10,
ICD10CM, RXNORM, LOINC, CPT) than docs/11_GLOSSARY.md's authoritative naming
section (SNOMED, LOINC, ICD10CM, RXNORM, CPT, UCUM). Reconciled here in
favour of the glossary -- bare "ICD10" dropped (ICD10CM is the system real
US EHR data actually carries), "UCUM" kept for ``Measurement.unit``-adjacent
use. docs/05_DATA_MODALITY_SPEC.md section 2.1 is updated in the same change
to stop describing the old, inconsistent set.
"""

from __future__ import annotations

# NOT behind TYPE_CHECKING -- every name below is a real Pydantic field type,
# resolved at class-definition time (the same rule already applied in
# clinical_text/schemas.py, core/provenance.py and core/config.py).
from datetime import date, datetime  # noqa: TC003
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from openbtk.core.schemas import CodeSystem, JsonValue
from openbtk.deid.schemas import DeidStatus


def _require_tz_aware(value: datetime | None) -> datetime | None:
    """Shared validator body for every timezone-aware datetime field below --
    the same real bug class ``ClinicalTextRecord.timestamp`` guards against:
    a naive clinical timestamp causes subtle, silent errors once timeline
    construction (``TemporalNormalizer``, task 6.4) starts comparing times
    across encounters, possibly across time zones."""
    if value is not None and value.tzinfo is None:
        raise ValueError(
            "must be timezone-aware; naive clinical timestamps cause real, "
            "subtle bugs in timeline construction (docs/05_DATA_MODALITY_SPEC.md "
            "section 2.1)."
        )
    return value


class Demographics(BaseModel):
    """A patient's demographic attributes, as loaded -- before or after
    de-identification (``PatientRecord.deid_status`` governs that, not this
    type).

    ``race``/``ethnicity`` are free text here, not an enum: FHIR's US Core
    extensions and OMOP's ``race_concept_id``/``ethnicity_concept_id`` both
    ultimately resolve to a categorical value from their own vocabularies
    (OMB categories, in practice), but OpenBTK does not bundle either
    vocabulary (docs/09_CODING_STANDARDS.md section 7 -- ``openbtk.terminology``
    is the right home for that resolution, and it is not built yet, M7).
    Loaders populate the human-readable display text they already have
    rather than inventing a premature enum.

    Example:
        >>> Demographics(gender="female").gender
        'female'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    birth_date: date | None = Field(None, description="Patient date of birth.")
    gender: Literal["male", "female", "other", "unknown"] | None = Field(
        None, description="Administrative gender, per FHIR's own value set."
    )
    race: str | None = Field(
        None, description="Display text, if the source provides it."
    )
    ethnicity: str | None = Field(
        None, description="Display text, if the source provides it."
    )
    deceased: bool = Field(
        False, description="Whether the patient is recorded as deceased."
    )
    deceased_date: date | None = Field(
        None, description="Date of death, if known and recorded."
    )


class Encounter(BaseModel):
    """A single care encounter (visit, admission, ...).

    ``encounter_id`` is the join key ``CodedEvent.encounter_ref`` and
    ``Measurement.encounter_ref`` point back to -- loaders are responsible
    for using the same, bare identifier on both sides (docs/05_DATA_MODALITY_SPEC.md
    section 2.1's own worked timeline example shows every event carrying its
    encounter inline for exactly this reason).

    Example:
        >>> Encounter(encounter_id="enc-1", encounter_type="inpatient").encounter_id
        'enc-1'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    encounter_id: str = Field(..., min_length=1)
    encounter_type: str | None = Field(
        None, description='e.g. "ambulatory", "inpatient", "emergency".'
    )
    start: datetime | None = Field(None, description="Timezone-aware.")
    end: datetime | None = Field(None, description="Timezone-aware.")
    status: str | None = Field(None, description='e.g. "finished", "in-progress".')

    @field_validator("start", "end")
    @classmethod
    def _validate_tz_aware(cls, value: datetime | None) -> datetime | None:
        return _require_tz_aware(value)


class CodedEvent(BaseModel):
    """A single coded clinical event: a condition, medication or procedure.

    Example:
        >>> from openbtk.core.schemas import CodeSystem
        >>> event = CodedEvent(code="385093006", system=CodeSystem.SNOMED)
        >>> event.system.value
        'SNOMED'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = Field(..., min_length=1)
    system: CodeSystem = Field(
        ..., description="The terminology system ``code`` is in."
    )
    display: str | None = Field(None)
    timestamp: datetime | None = Field(None, description="Timezone-aware.")
    encounter_ref: str | None = Field(
        None, description="The ``Encounter.encounter_id`` this event occurred within."
    )
    status: str | None = Field(
        None, description='e.g. "active", "resolved", "completed".'
    )

    @field_validator("timestamp")
    @classmethod
    def _validate_tz_aware(cls, value: datetime | None) -> datetime | None:
        return _require_tz_aware(value)


class Measurement(BaseModel):
    """A single observation or lab result.

    ``value`` is ``float | str | None`` because not every observation is
    numeric -- a qualitative result ("Positive", "Detected") is real,
    common lab data, not an edge case to reject.

    ``display`` is a deliberate, disclosed addition beyond
    docs/05_DATA_MODALITY_SPEC.md section 2.1's original field list (which
    otherwise matches exactly) -- ``PatientTimelineSerializer`` (task 6.5)
    needs a human-readable name ("WBC") to render a lab result at all; a
    bare LOINC code is not something a clinician or an LLM reading a
    rendered timeline can interpret. ``CodedEvent`` already had this field;
    ``Measurement`` lacking it was an oversight in the original spec, fixed
    here rather than shipping a serializer that can only ever show raw
    codes.

    Example:
        >>> m = Measurement(code="6690-2", display="WBC", value=14.2, unit="10*3/uL")
        >>> m.system.value
        'LOINC'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = Field(..., min_length=1)
    system: CodeSystem = Field(CodeSystem.LOINC)
    display: str | None = Field(None)
    value: float | str | None = Field(None)
    unit: str | None = Field(None, description="UCUM unit string, if known.")
    reference_range: tuple[float, float] | None = Field(
        None, description="(low, high), if the source reports one."
    )
    timestamp: datetime | None = Field(None, description="Timezone-aware.")
    encounter_ref: str | None = Field(
        None, description="The ``Encounter.encounter_id`` this observation belongs to."
    )

    @field_validator("timestamp")
    @classmethod
    def _validate_tz_aware(cls, value: datetime | None) -> datetime | None:
        return _require_tz_aware(value)


class PatientRecord(BaseModel):
    """One patient's structured clinical record, source-agnostic.

    Example:
        >>> record = PatientRecord(
        ...     patient_id="pt-1",
        ...     demographics=Demographics(gender="female"),
        ...     source_system="fhir-r4",
        ... )
        >>> record.deid_status
        <DeidStatus.UNKNOWN: 'unknown'>
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    patient_id: str = Field(
        ..., min_length=1, description="Stable identifier, pseudonymous."
    )
    demographics: Demographics = Field(...)
    encounters: list[Encounter] = Field(default_factory=list)
    conditions: list[CodedEvent] = Field(default_factory=list)
    medications: list[CodedEvent] = Field(default_factory=list)
    procedures: list[CodedEvent] = Field(default_factory=list)
    observations: list[Measurement] = Field(default_factory=list)
    source_system: str = Field(
        ..., min_length=1, description='e.g. "fhir-r4", "omop-cdm-5.4".'
    )
    deid_status: DeidStatus = Field(DeidStatus.UNKNOWN)
    metadata: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Loader-specific extras (e.g. the source bundle id) -- "
        "same convention as ClinicalTextRecord.metadata.",
    )
