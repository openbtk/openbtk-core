"""EHR guardrails (M6 task 7.6, docs/05_DATA_MODALITY_SPEC.md section 2.5):
code validity, referential integrity, unit plausibility.

**Cohort de-identification (k-anonymity)**, the fourth guardrail section
2.5 names, is deliberately not built here: k-anonymity over quasi-
identifiers on export is a real, separate feature (choosing and
generalising quasi-identifier fields, a suppression/generalisation
strategy) that this task's own roadmap entry ("code validity, referential
integrity, units") does not list -- building it unasked would be exactly
the unscoped, unverified extra work CLAUDE.md's discipline warns against.

**Guardrails live at L3** (docs/03_ARCHITECTURE.md section 8.3) and may
import ``openbtk.data.ehr`` (L2) downward -- these are the first
guardrails that duck-type against a specific modality's schema shape
(``PatientRecord``'s own field names), which is fine: "composable across
modalities" (FR-G-08) means a guardrail is not *owned* by a modality
module, not that it may never know a modality's shape at all.
"""

from __future__ import annotations

from typing import Any

from openbtk.core.base import BaseGuardrail
from openbtk.core.registry import GUARDRAIL_REGISTRY
from openbtk.core.schemas import GuardrailResult, GuardrailSeverity
from openbtk.guardrails.terminology_validity import (
    TerminologyValidityGuardrail,
    _looks_like_patient_record,
)

_EVENT_FIELDS = ("conditions", "medications", "procedures")
_MEASUREMENT_FIELD = "observations"

# A small, illustrative allowlist of common UCUM unit strings actually
# seen in clinical measurements -- NOT a full UCUM grammar validator
# (UCUM's real syntax supports prefixes, exponents and unit algebra this
# does not parse). Disclosed, bounded scope, the same treatment already
# given to BundledMinimalBackend's ICD-10-CM subset.
_KNOWN_UCUM_UNITS = frozenset(
    {
        "%",
        "mg/dL",
        "g/dL",
        "mmol/L",
        "mEq/L",
        "mmHg",
        "/min",
        "mL/min",
        "10*3/uL",
        "10*6/uL",
        "10*9/L",
        "U/L",
        "IU/L",
        "ng/mL",
        "pg/mL",
        "ug/mL",
        "mg/mL",
        "mmol/mol",
        "kg",
        "g",
        "mg",
        "ug",
        "cm",
        "mm",
        "mL",
        "L",
        "degC",
        "degF",
        "/uL",
        "/L",
        "s",
        "min",
        "h",
    }
)

# A handful of extremely common labs' wide, illustrative adult reference
# bounds -- (low, high, expected UCUM unit), verified LOINC codes (see
# module docstring in openbtk.terminology.bundled for how "verified"
# is defined here; each of these was checked the same way, against the
# real NLM Clinical Table Search Service, not recalled from training
# data). NOT a clinical decision support tool: bounds are deliberately
# wide, to catch gross data-entry errors (an order-of-magnitude typo, a
# unit mismatch) rather than to flag real, clinically abnormal results.
_PHYSIOLOGICAL_RANGES: dict[str, tuple[float, float, str]] = {
    "6690-2": (2.0, 20.0, "10*3/uL"),  # Leukocytes [WBC]
    "718-7": (5.0, 22.0, "g/dL"),  # Hemoglobin
    "777-3": (10.0, 800.0, "10*3/uL"),  # Platelets
    "2951-2": (110.0, 160.0, "mmol/L"),  # Sodium
    "2823-3": (2.0, 7.5, "mmol/L"),  # Potassium
    "2345-7": (30.0, 500.0, "mg/dL"),  # Glucose
    "2160-0": (0.1, 10.0, "mg/dL"),  # Creatinine
}


@GUARDRAIL_REGISTRY.register("guardrail.ehr.code_validity")
class EHRCodeValidityGuardrail(TerminologyValidityGuardrail):
    """Every code in a ``PatientRecord`` (conditions/medications/
    procedures/observations) exists in its declared system.

    Genuinely the identical check :class:`TerminologyValidityGuardrail`
    already performs -- ``PatientRecord`` was already one of its
    supported payload shapes. Registered under this separate, modality-
    scoped key because docs/05_DATA_MODALITY_SPEC.md section 2.5 names
    one explicitly, and a real subclass (not
    ``TERMINOLOGY_REGISTRY.register_alias``, whose own docstring scopes
    it to a deprecation transition for the same class under a renamed
    key, not two permanent, independent semantic roles) keeps
    ``registry_key``/provenance honest about which key actually
    constructed a given instance.
    """


@GUARDRAIL_REGISTRY.register("guardrail.ehr.referential")
class ReferentialIntegrityGuardrail(BaseGuardrail):
    """Every event's ``encounter_ref`` resolves to a real encounter, and
    its timestamp (when known) falls within that encounter's window.

    Example:
        >>> import contextlib, io
        >>> from datetime import datetime, UTC
        >>> from openbtk.core.schemas import CodeSystem
        >>> with contextlib.redirect_stdout(io.StringIO()):
        ...     from openbtk.data.ehr.schemas import (
        ...         CodedEvent, Demographics, Encounter, PatientRecord,
        ...     )
        >>> record = PatientRecord(
        ...     patient_id="pt-1",
        ...     demographics=Demographics(),
        ...     encounters=[Encounter(
        ...         encounter_id="enc-1",
        ...         start=datetime(2024, 3, 14, tzinfo=UTC),
        ...         end=datetime(2024, 3, 15, tzinfo=UTC),
        ...     )],
        ...     conditions=[CodedEvent(
        ...         code="x", system=CodeSystem.SNOMED, encounter_ref="enc-1",
        ...         timestamp=datetime(2024, 3, 20, tzinfo=UTC),
        ...     )],
        ...     source_system="fhir-r4",
        ... )
        >>> ReferentialIntegrityGuardrail().check(record).passed
        False
    """

    def check(self, payload: Any) -> GuardrailResult:
        if not _looks_like_patient_record(payload):
            return GuardrailResult(
                passed=True,
                severity=GuardrailSeverity.INFO,
                guardrail_key=self.registry_key,
                message="Not a PatientRecord-shaped payload.",
            )
        encounters_by_id = {e.encounter_id: e for e in payload.encounters}
        violations: list[str] = []
        for event in _all_events(payload):
            if event.encounter_ref is None:
                continue
            encounter = encounters_by_id.get(event.encounter_ref)
            if encounter is None:
                violations.append(f"dangling encounter_ref {event.encounter_ref!r}")
                continue
            if event.timestamp is None:
                continue
            if encounter.start is not None and event.timestamp < encounter.start:
                violations.append(
                    f"event before encounter {event.encounter_ref!r} start"
                )
            if encounter.end is not None and event.timestamp > encounter.end:
                violations.append(f"event after encounter {event.encounter_ref!r} end")
        if violations:
            return GuardrailResult(
                passed=False,
                severity=GuardrailSeverity.BLOCK,
                guardrail_key=self.registry_key,
                message=f"{len(violations)} referential integrity violation(s).",
                details={"violations": violations[:20], "count": len(violations)},
            )
        return GuardrailResult(
            passed=True,
            severity=GuardrailSeverity.INFO,
            guardrail_key=self.registry_key,
            message="All event references and timestamps are consistent.",
        )


@GUARDRAIL_REGISTRY.register("guardrail.ehr.units")
class UnitPlausibilityGuardrail(BaseGuardrail):
    """Every observation's unit is a recognised UCUM string, and its
    value falls within a wide, illustrative physiological range for
    known LOINC codes.

    A data-quality check, not a clinical decision support tool -- see
    this module's own ``_PHYSIOLOGICAL_RANGES`` docstring comment for the
    real, disclosed scope.

    Example:
        >>> from openbtk.data.ehr.schemas import (
        ...     Demographics, Measurement, PatientRecord,
        ... )
        >>> record = PatientRecord(
        ...     patient_id="pt-1",
        ...     demographics=Demographics(),
        ...     observations=[Measurement(code="6690-2", value=400000, unit="10*3/uL")],
        ...     source_system="fhir-r4",
        ... )
        >>> UnitPlausibilityGuardrail().check(record).passed
        False
    """

    def check(self, payload: Any) -> GuardrailResult:
        if not _looks_like_patient_record(payload):
            return GuardrailResult(
                passed=True,
                severity=GuardrailSeverity.INFO,
                guardrail_key=self.registry_key,
                message="Not a PatientRecord-shaped payload.",
            )
        violations: list[str] = []
        for observation in getattr(payload, _MEASUREMENT_FIELD):
            if (
                observation.unit is not None
                and observation.unit not in _KNOWN_UCUM_UNITS
            ):
                violations.append(f"unrecognised unit {observation.unit!r}")
            expected = _PHYSIOLOGICAL_RANGES.get(observation.code)
            if expected is None or not isinstance(observation.value, (int, float)):
                continue
            low, high, _unit = expected
            if not (low <= observation.value <= high):
                violations.append(
                    f"{observation.code} value {observation.value} outside "
                    f"[{low}, {high}]"
                )
        if violations:
            return GuardrailResult(
                passed=False,
                severity=GuardrailSeverity.WARNING,
                guardrail_key=self.registry_key,
                message=f"{len(violations)} unit/range implausibility(ies).",
                details={"violations": violations[:20], "count": len(violations)},
            )
        return GuardrailResult(
            passed=True,
            severity=GuardrailSeverity.INFO,
            guardrail_key=self.registry_key,
            message="All observation units and values are plausible.",
        )


def _all_events(payload: Any) -> list[Any]:
    events: list[Any] = []
    for field in _EVENT_FIELDS:
        events.extend(getattr(payload, field))
    events.extend(getattr(payload, _MEASUREMENT_FIELD))
    return events
