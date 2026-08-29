"""
Pydantic schemas for the clinical text modality.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from opentbtk.core.schemas import LinkedEntity


class ClinicalTextRecord(BaseModel):
    """A single clinical text document (note, report, summary)."""

    model_config = {"frozen": True}

    record_id: str = Field(..., description="Unique identifier (never raw MRN).")
    source: str = Field(..., description="Source name e.g. 'mimic-iv-note', 'plain_text'.")
    note_type: str | None = Field(None, description="E.g. 'Discharge Summary', 'Radiology Report'.")
    raw_text: str = Field(..., min_length=1, description="Full note text (may contain PHI — deidentify before storage).")
    sections: dict[str, str] | None = Field(None, description="Section label → section text, populated by SectionSegmenter.")
    language: str = Field("en", description="ISO 639-1 language code.")
    created_at: datetime | None = Field(None, description="Note creation timestamp (de-identified if PHI).")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Non-PHI metadata: note_type, encounter_id (hashed), etc.")


class ClinicalTextChunk(BaseModel):
    """A chunk derived from a ClinicalTextRecord."""

    model_config = {"frozen": True}

    chunk_id: str = Field(..., description="UUID of this chunk.")
    record_id: str = Field(..., description="Parent ClinicalTextRecord identifier.")
    text: str = Field(..., min_length=1)
    section: str | None = Field(None, description="Clinical section label if chunk is section-bounded.")
    chunk_index: int = Field(..., ge=0, description="Position of this chunk within the record.")
    token_count: int = Field(..., ge=1)
    char_start: int = Field(..., ge=0, description="Character start offset in original raw_text.")
    char_end: int = Field(..., ge=1, description="Character end offset in original raw_text.")
    entities: list[LinkedEntity] = Field(default_factory=list, description="Extracted clinical entities with vocabulary links.")
    metadata: dict[str, Any] = Field(default_factory=dict)
