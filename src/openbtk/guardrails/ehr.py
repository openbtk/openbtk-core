"""EHR guardrails (M6 task 7.6, docs/05_DATA_MODALITY_SPEC.md section 2.5):
code validity, referential integrity, unit plausibility.

**Cohort de-identification (k-anonymity)**, the fourth guardrail section
2.5 names, is ``guardrail.ehr.k_anonymity`` (FR-D-11). It was left out of the
first three because choosing quasi-identifiers and a generalisation strategy
is a feature of its own; that machinery now lives in
``openbtk.deid.kanonymity`` and this guardrail is its check over a cohort.

**Guardrails live at L3** (docs/03_ARCHITECTURE.md section 8.3) and may
import ``openbtk.data.ehr`` (L2) downward -- these are the first
guardrails that duck-type against a specific modality's schema shape
(``PatientRecord``'s own field names), which is fine: "composable across
modalities" (FR-G-08) means a guardrail is not *owned* by a modality
module, not that it may never know a modality's shape at all.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from typing import TYPE_CHECKING, Any

from openbtk.core.base import BaseGuardrail
from openbtk.core.registry import GUARDRAIL_REGISTRY
from openbtk.core.schemas import GuardrailResult, GuardrailSeverity
from openbtk.data.ehr.cohort import QUASI_IDENTIFIER_FIELDS, quasi_identifiers
from openbtk.deid.kanonymity import k_anonymity_report
from openbtk.guardrails.terminology_validity import (
    TerminologyValidityGuardrail,
    _looks_like_patient_record,
)

if TYPE_CHECKING:
    from openbtk.core.provenance import ComponentProvenance

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


@GUARDRAIL_REGISTRY.register("guardrail.ehr.k_anonymity")
class CohortKAnonymityGuardrail(BaseGuardrail):
    """A cohort is k-anonymous over its quasi-identifiers, or the check warns.

    Give it the cohort (any iterable of ``PatientRecord``, consumed once) and it counts
    how many patients share each combination of the chosen attributes. If the smallest
    group is under ``k`` the result is a ``WARNING`` (per docs/06_SECURITY_COMPLIANCE.md
    section 3.10) saying how many patients are in a too-small group and how many are
    unique. The result carries **counts only**, never a value.

    Args:
        k: The smallest acceptable group. ``5`` is a common floor, not a rule; the right
            value is a policy decision for your data.
        quasi_identifiers: Attributes to combine; see
            ``openbtk.data.ehr.cohort.QUASI_IDENTIFIER_FIELDS``.
        as_of: The date ``age`` is computed on (required only when ``age`` is used).

    It reaches "satisfied" only over the attributes you name, and says nothing about
    what a group has in common. See ``openbtk.deid.kanonymity`` for what k-anonymity
    does and does not give you, and ``anonymise_to_k`` to generalise a table until it
    passes.

    Example:
        >>> from openbtk.data.ehr.schemas import Demographics, PatientRecord
        >>> cohort = [
        ...     PatientRecord(
        ...         patient_id=f"pt-{i}",
        ...         demographics=Demographics(gender="female"),
        ...         source_system="fhir-r4",
        ...     )
        ...     for i in range(6)
        ... ]
        >>> guardrail = CohortKAnonymityGuardrail(k=5, quasi_identifiers=["gender"])
        >>> guardrail.check(cohort).passed
        True
        >>> guardrail.check(cohort[:2]).passed
        False
    """

    def __init__(
        self,
        *,
        k: int = 5,
        quasi_identifiers: Sequence[str] = (
            "birth_year",
            "gender",
            "race",
            "ethnicity",
        ),
        as_of: date | str | None = None,
    ) -> None:
        self._k = k
        self._fields = list(quasi_identifiers)
        self._as_of = as_of

    def provenance(self) -> ComponentProvenance:
        """Records the threshold and the attributes used."""
        return (
            super()
            .provenance()
            .model_copy(
                update={
                    "config": {
                        "k": self._k,
                        "quasi_identifiers": list(self._fields),
                        "as_of": str(self._as_of) if self._as_of else None,
                    }
                }
            )
        )

    def _result(
        self,
        passed: bool,
        severity: GuardrailSeverity,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> GuardrailResult:
        return GuardrailResult(
            passed=passed,
            severity=severity,
            guardrail_key=self.registry_key,
            message=message,
            details=details or {},
        )

    def check(self, payload: Any) -> GuardrailResult:
        if isinstance(payload, (str, bytes, Mapping)) or not isinstance(
            payload, Iterable
        ):
            return self._result(True, GuardrailSeverity.INFO, "Not a cohort.")
        records = iter(payload)
        first = next(records, None)
        if first is None:
            return self._result(True, GuardrailSeverity.INFO, "The cohort is empty.")
        if not _looks_like_patient_record(first):
            return self._result(
                True, GuardrailSeverity.INFO, "Not a cohort of PatientRecords."
            )
        try:
            if self._k < 1:
                raise ValueError("k must be at least 1")
            as_of = (
                date.fromisoformat(self._as_of)
                if isinstance(self._as_of, str)
                else self._as_of
            )
            rows = quasi_identifiers(
                itertools.chain([first], records), self._fields, as_of=as_of
            )
            report = k_anonymity_report(rows, self._fields, k=self._k)
        except ValueError as e:
            return self._result(
                False,
                GuardrailSeverity.WARNING,
                f"k-anonymity was not checked: {e}",
                {"available": list(QUASI_IDENTIFIER_FIELDS)},
            )
        details = report.model_dump(mode="json")
        if report.satisfied:
            return self._result(
                True,
                GuardrailSeverity.INFO,
                f"Every group has at least {self._k} patients "
                f"(smallest: {report.k_achieved}).",
                details,
            )
        return self._result(
            False,
            GuardrailSeverity.WARNING,
            f"The smallest group has {report.k_achieved} patient(s), under "
            f"k={self._k}: "
            f"{report.n_below_k} of {report.n_records} patients are in a group that "
            f"is too small, {report.n_unique} of them unique.",
            details,
        )
