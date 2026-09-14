"""Clinical text record and chunk schemas (docs/05_DATA_MODALITY_SPEC.md
section 1.1).

**``sections`` maps to spans, not strings.** v1 used ``dict[str, str]``,
duplicating the note text per section -- memory waste and a second copy of
PHI to track. Spans index into ``text`` instead.

**``ClinicalTextChunk.span`` is mandatory.** Every chunk knows its exact
offsets in the parent record, which is what makes a verifiable citation
possible later (docs/04_API_DESIGN.md section 4's ``GroundednessGuardrail``).

**``deid_status`` is explicit, not inferred.** ``DeidStatus`` lives in
``openbtk.deid.schemas``, not here -- both ``clinical_text`` and ``ehr``
need it, and the layering rule keeps the two modalities independent of
each other (docs/03_ARCHITECTURE.md section 2), so a concept both need
belongs in a module both already depend on downward. Making PHI state
part of the type means a pipeline step can refuse to send a ``RAW`` record
to an off-site provider by checking one field, rather than trusting that
de-identification happened somewhere upstream.
"""

from __future__ import annotations

# NOT behind TYPE_CHECKING: every name below is used as a real Pydantic
# field type, resolved at class-definition time -- the same mistake already
# made and fixed once in core/provenance.py, core/config.py and
# deid/schemas.py. Hiding these imports raises a Pydantic annotation-
# resolution error at the first import of this module, not merely at first
# use.
from datetime import datetime  # noqa: TC003

from pydantic import BaseModel, ConfigDict, Field, field_validator

from openbtk.core.schemas import JsonValue, LinkedEntity, TextSpan  # noqa: TC001
from openbtk.deid.schemas import DeidStatus


class ClinicalTextRecord(BaseModel):
    """One clinical note, as loaded -- before or after de-identification.

    Example:
        >>> record = ClinicalTextRecord(
        ...     record_id="note-1",
        ...     source="synthea",
        ...     text="Chief Complaint: chest pain.",
        ... )
        >>> record.deid_status
        <DeidStatus.UNKNOWN: 'unknown'>
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str = Field(
        ..., min_length=1, description="Stable identifier, hashed if derived from PHI."
    )
    source: str = Field(
        ..., min_length=1, description='e.g. "mimic-iv-note", "synthea".'
    )
    text: str = Field(...)
    note_type: str | None = Field(
        None, description='e.g. "Discharge Summary", "Radiology".'
    )
    sections: dict[str, TextSpan] | None = Field(
        None, description="Section label -> span in `text`."
    )
    patient_ref: str | None = Field(None, description="Hashed/pseudonymous.")
    encounter_ref: str | None = Field(None)
    timestamp: datetime | None = Field(
        None,
        description=(
            "Timezone-aware. Ruff's DTZ rule catches a naive datetime() call "
            "in OUR OWN code; this field's own validator is the runtime "
            "backstop for a naive value arriving from a caller or an "
            "external data source (a loader parsing a CSV column, say), "
            "which ruff cannot see."
        ),
    )
    deid_status: DeidStatus = Field(DeidStatus.UNKNOWN)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("timestamp")
    @classmethod
    def _reject_naive_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError(
                "timestamp must be timezone-aware; naive clinical timestamps "
                "cause real, subtle bugs in timeline construction "
                "(docs/05_DATA_MODALITY_SPEC.md section 1.1)."
            )
        return value


class ClinicalTextChunk(BaseModel):
    """One retrieval- or embedding-ready unit of a ``ClinicalTextRecord``.

    Example:
        >>> chunk = ClinicalTextChunk(
        ...     chunk_id="note-1:0",
        ...     record_id="note-1",
        ...     text="Chief Complaint: chest pain.",
        ...     span=TextSpan(start=0, end=29, label="chunk", confidence=1.0),
        ...     token_count=7,
        ... )
        >>> chunk.token_count
        7
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(..., min_length=1)
    record_id: str = Field(..., min_length=1)
    text: str = Field(...)
    span: TextSpan = Field(..., description="Offsets into the parent record's text.")
    section: str | None = Field(None)
    token_count: int = Field(..., ge=1, description="Exact subword count.")
    entities: list[LinkedEntity] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
