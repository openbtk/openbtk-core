"""
openbtk shared schemas used across modules.

Kept separate from base.py to avoid circular imports. Import from here
when you need GuardrailResult, LinkedEntity, or other cross-cutting types.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class GuardrailSeverity(str, Enum):
    """Severity level of a guardrail finding."""
    INFO = "info"           # Informational — no action required
    WARNING = "warning"     # Flagged — log and continue
    BLOCK = "block"         # Violation — must not proceed


class GuardrailResult(BaseModel):
    """Result of a guardrail check."""

    model_config = {"frozen": True}

    passed: bool = Field(..., description="True if the check passed (no violation).")
    severity: GuardrailSeverity = Field(
        ..., description="Severity of the finding."
    )
    guardrail_name: str = Field(
        ..., description="Name/key of the guardrail that produced this result."
    )
    message: str = Field(..., description="Human-readable result description.")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured details about the finding (e.g., detected PHI spans).",
    )


class LinkedEntity(BaseModel):
    """A clinical concept entity extracted from text and linked to a vocabulary."""

    model_config = {"frozen": True}

    text: str = Field(..., description="Surface form of the entity in source text.")
    start: int = Field(..., description="Character start offset in source text.")
    end: int = Field(..., description="Character end offset in source text.")
    label: str = Field(..., description="Entity type label (e.g., 'DISEASE', 'DRUG').")
    cui: str | None = Field(None, description="UMLS Concept Unique Identifier.")
    snomed_code: str | None = Field(None, description="SNOMED CT concept ID.")
    icd10_code: str | None = Field(None, description="ICD-10 code if applicable.")
    loinc_code: str | None = Field(None, description="LOINC code if applicable.")
    rxnorm_code: str | None = Field(None, description="RxNorm CUI if applicable.")
    confidence: float | None = Field(None, ge=0.0, le=1.0)
