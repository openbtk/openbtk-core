"""De-identification schemas: what a detection is, and what a run reports.

docs/04_API_DESIGN.md section 5 is authoritative for the public shape.
ADR-0006 is authoritative for the *design decisions* behind it -- read that
first if you're extending this module, not just this docstring.

**``Detection`` deliberately omits the detected text.** A report containing
the PHI it found would itself be a PHI document, defeating its own purpose
(docs/06_SECURITY_COMPLIANCE.md section 3.8, T8). Every schema in this
module is safe to archive, log (at INFO, not DEBUG-with-payload), forward to
a privacy officer, or attach to a run manifest, by construction -- there is
no field anywhere here that can hold document content.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# NOT behind TYPE_CHECKING: Detection.span uses this as a real Pydantic field
# type, resolved at class-definition time -- the same mistake already made
# and fixed once in core/provenance.py and core/config.py. Hiding this
# import raises a Pydantic annotation-resolution error at the first import
# of this module, not merely at first use.
from openbtk.core.schemas import TextSpan  # noqa: TC001


class PHICategory(StrEnum):
    """The 18 HIPAA Safe Harbor identifier categories (45 CFR
    164.514(b)(2)) -- a public, regulatory enumeration, not something
    specific to any patient or institution.

    Two categories cannot appear in *text* at all and are included here
    only for completeness of the legal enumeration, not because a text
    recognizer can ever populate them:

    - ``FULL_FACE_PHOTO``: an imaging artefact. Detecting it is out of
      scope for `openbtk.deid` on text input, and out of scope for OpenBTK
      entirely under the scope gate (CLAUDE.md rule 13) until an imaging
      modality ships.
    - ``BIOMETRIC_IDENTIFIER``: fingerprints, retinal scans, voiceprints --
      likewise not a text-detectable category.

    A recognizer that claims to detect either of these on text input should
    be treated as a bug, not a feature.
    """

    NAME = "name"
    GEOGRAPHIC_SUBDIVISION = "geographic_subdivision"
    DATE = "date"
    PHONE_NUMBER = "phone_number"
    FAX_NUMBER = "fax_number"
    EMAIL = "email"
    SSN = "ssn"
    MEDICAL_RECORD_NUMBER = "medical_record_number"
    HEALTH_PLAN_BENEFICIARY_NUMBER = "health_plan_beneficiary_number"
    ACCOUNT_NUMBER = "account_number"
    CERTIFICATE_LICENSE_NUMBER = "certificate_license_number"
    VEHICLE_IDENTIFIER = "vehicle_identifier"
    DEVICE_IDENTIFIER = "device_identifier"
    URL = "url"
    IP_ADDRESS = "ip_address"
    BIOMETRIC_IDENTIFIER = "biometric_identifier"
    FULL_FACE_PHOTO = "full_face_photo"
    OTHER_UNIQUE_IDENTIFIER = "other_unique_identifier"


class DeidMode(StrEnum):
    """How a detected span is transformed. See ``openbtk.deid.transforms``."""

    REDACT = "redact"
    SURROGATE = "surrogate"
    HASH = "hash"
    TAG = "tag"
    DATE_SHIFT = "date_shift"


class DeidStatus(StrEnum):
    """What state a record's PHI is in -- lives here, not in any one
    modality's own schemas module, because both ``clinical_text`` and
    ``ehr`` need it (docs/05_DATA_MODALITY_SPEC.md sections 1.1 and 2.1)
    and the layering rule (docs/03_ARCHITECTURE.md section 2) keeps the two
    modalities independent of each other -- a shared concept belongs in a
    module both already depend on downward, not in either one's own
    package.

    Making PHI state part of the type (docs/05_DATA_MODALITY_SPEC.md
    section 1.1) is cheap and prevents a whole class of accident: a
    pipeline step can refuse to send a ``RAW`` record to an off-site
    provider by checking one field, rather than trusting that de-id
    happened somewhere upstream.
    """

    UNKNOWN = "unknown"
    RAW = "raw"
    DEIDENTIFIED = "deidentified"
    SURROGATE = "surrogate"


class Detection(BaseModel):
    """One PHI span found by a recognizer or the merger.

    Example:
        >>> from openbtk.core.schemas import TextSpan
        >>> d = Detection(
        ...     category=PHICategory.SSN,
        ...     span=TextSpan(start=10, end=21, label="ssn", confidence=0.99),
        ...     confidence=0.99,
        ...     method="rule",
        ... )
        >>> d.category
        <PHICategory.SSN: 'ssn'>

    Note:
        ``span.confidence`` and ``confidence`` are kept equal by every
        producer in this codebase -- ``TextSpan`` is a shared generic type
        (also used by guardrails and entity linking) that always carries its
        own confidence field, not a second, independent estimate. Keeping
        one authoritative value and mirroring it into the span avoids two
        numbers silently drifting apart.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    category: PHICategory = Field(..., description="Which Safe Harbor category.")
    span: TextSpan = Field(..., description="Character offsets within the source text.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detector confidence.")
    method: Literal["rule", "ner", "llm_verifier", "ensemble"] = Field(
        ..., description="Which recognizer (or the merger) produced this."
    )


class RiskEstimate(BaseModel):
    """A coarse, explainable residual-risk judgement for one document.

    Not specified beyond a bare type reference in docs/04_API_DESIGN.md
    section 5 -- designed from scratch. Deliberately a bounded category plus
    a human-readable reason rather than a numeric score: a privacy officer
    reading a ``DeidReport`` needs "why", and a single float invites false
    precision this milestone has no calibration data to back up.

    Example:
        >>> RiskEstimate(level="low", rationale="No low-confidence spans.").level
        'low'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    level: Literal["low", "medium", "high"] = Field(
        ..., description="Coarse residual re-identification risk."
    )
    rationale: str = Field(
        ..., min_length=1, description="Human-readable basis for the level."
    )


class DeidReport(BaseModel):
    """Per-document audit output of one ``DeidEngine.deidentify()`` call.

    Safe to archive, forward to a privacy officer, or attach to a run
    manifest -- see this module's docstring for why no field here can ever
    hold document content.

    Example:
        >>> report = DeidReport(
        ...     document_id="doc-1",
        ...     entity_counts={PHICategory.SSN: 1},
        ...     detections=[],
        ...     residual_risk=RiskEstimate(level="low", rationale="clean"),
        ...     engine_config_hash="abc123",
        ... )
        >>> report.entity_counts[PHICategory.SSN]
        1
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(
        ..., min_length=1, description="Caller-supplied identifier."
    )
    entity_counts: dict[PHICategory, int] = Field(
        ..., description="Counts per category -- never the values themselves."
    )
    detections: list[Detection] = Field(
        ..., description="Every span found, post-merge."
    )
    residual_risk: RiskEstimate = Field(
        ..., description="Coarse residual-risk judgement."
    )
    engine_config_hash: str = Field(
        ...,
        min_length=1,
        description="Hash of the engine configuration that produced this report.",
    )


class DeidResult(BaseModel):
    """The return value of ``DeidEngine.deidentify()``: transformed text plus
    its audit report.

    Example:
        >>> report = DeidReport(
        ...     document_id="doc-1",
        ...     entity_counts={},
        ...     detections=[],
        ...     residual_risk=RiskEstimate(level="low", rationale="clean"),
        ...     engine_config_hash="abc123",
        ... )
        >>> DeidResult(text="hello", report=report).text
        'hello'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(..., description="The de-identified text.")
    report: DeidReport = Field(..., description="The audit report for this run.")
