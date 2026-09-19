"""``TerminologyValidityGuardrail``: every emitted code exists in its
declared system (FR-G-03).

Accepts several real payload shapes rather than one rigid type, since
"a code that needs validating" shows up in more than one place: a single
object with ``.code``/``.system`` attributes (``CodedEvent``,
``Measurement`` -- duck-typed, not an isinstance check, so it also works
for any future modality's own coded-event type without this guardrail
needing to import it), a bare ``(code, system)`` tuple, an iterable of
either, or a whole ``PatientRecord`` (every code across its conditions/
medications/procedures/observations is checked at once -- this is what
``guardrail.ehr.code_validity`` reuses this exact class for, see
``openbtk.guardrails.ehr``). Any other payload shape passes with an INFO
result noting nothing to check: this guardrail's job is to validate
codes it can find, not to reject payloads unrelated to its concern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openbtk.core.base import BaseGuardrail
from openbtk.core.errors import TerminologyError
from openbtk.core.registry import GUARDRAIL_REGISTRY
from openbtk.core.schemas import CodeSystem, GuardrailResult, GuardrailSeverity
from openbtk.terminology.bundled import BundledMinimalBackend

if TYPE_CHECKING:
    from openbtk.core.base import BaseTerminologyService

_PATIENT_RECORD_EVENT_FIELDS = (
    "conditions",
    "medications",
    "procedures",
    "observations",
)


def _as_code_system_pair(item: Any) -> tuple[str, CodeSystem] | None:
    if (
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], str)
        and isinstance(item[1], CodeSystem)
    ):
        return item[0], item[1]
    code = getattr(item, "code", None)
    system = getattr(item, "system", None)
    if isinstance(code, str) and isinstance(system, CodeSystem):
        return code, system
    return None


def _looks_like_patient_record(payload: Any) -> bool:
    return all(hasattr(payload, field) for field in _PATIENT_RECORD_EVENT_FIELDS)


def _extract_code_system_pairs(payload: Any) -> list[tuple[str, CodeSystem]]:
    direct = _as_code_system_pair(payload)
    if direct is not None:
        return [direct]
    if _looks_like_patient_record(payload):
        items: list[Any] = []
        for field in _PATIENT_RECORD_EVENT_FIELDS:
            items.extend(getattr(payload, field))
    elif isinstance(payload, (list, tuple)):
        items = list(payload)
    else:
        return []
    pairs = (_as_code_system_pair(item) for item in items)
    return [p for p in pairs if p is not None]


@GUARDRAIL_REGISTRY.register("guardrail.general.terminology")
class TerminologyValidityGuardrail(BaseGuardrail):
    """Checks that every code found in a payload validates against its
    declared terminology system.

    Args:
        terminology: The service to validate codes against. Defaults to
            :class:`~openbtk.terminology.bundled.BundledMinimalBackend` --
            a real, working, zero-config default so this guardrail is
            usable and contract-testable with no required constructor
            args, at the cost of only covering ICD-10-CM out of the box.
            Inject a more complete backend (``UMLSRestBackend``,
            ``LocalVocabBackend``, or ``CachedTerminologyService``
            wrapping either) for real coverage of other systems.

    A code the backend cannot confirm (``TerminologyError`` -- an
    unlicensed or unavailable backend) is reported as a WARNING, not
    silently treated as valid or as a BLOCK-worthy invalid code:
    docs/03_ARCHITECTURE.md section 8.2 is explicit that a terminology
    backend "does not silently return valid," and conflating "could not
    check" with "is invalid" would be its own kind of silent wrong answer.

    Example:
        >>> from openbtk.core.schemas import CodeSystem
        >>> guardrail = TerminologyValidityGuardrail()
        >>> guardrail.check(("E11.9", CodeSystem.ICD10CM)).passed
        True
        >>> guardrail.check(("Z99.999", CodeSystem.ICD10CM)).passed
        False
    """

    def __init__(self, *, terminology: BaseTerminologyService | None = None) -> None:
        self._terminology = (
            terminology if terminology is not None else BundledMinimalBackend()
        )

    def check(self, payload: Any) -> GuardrailResult:
        pairs = _extract_code_system_pairs(payload)
        if not pairs:
            return GuardrailResult(
                passed=True,
                severity=GuardrailSeverity.INFO,
                guardrail_key=self.registry_key,
                message="No codes to validate in this payload.",
            )
        invalid: list[str] = []
        errored: list[str] = []
        for code, system in pairs:
            label = f"{system.value}:{code}"
            try:
                valid = self._terminology.validate(code, system)
            except TerminologyError:
                errored.append(label)
                continue
            if valid:
                continue
            # A backend that cannot say "does not exist" (a partial subset)
            # has not found the code INVALID -- it has failed to confirm it.
            if self._terminology.is_authoritative(system):
                invalid.append(label)
            else:
                errored.append(label)
        if invalid:
            return GuardrailResult(
                passed=False,
                severity=GuardrailSeverity.BLOCK,
                guardrail_key=self.registry_key,
                message=(
                    f"{len(invalid)} of {len(pairs)} code(s) do not exist in "
                    "their declared system."
                ),
                details={"invalid": invalid, "unverifiable": errored},
            )
        if errored:
            return GuardrailResult(
                passed=False,
                severity=GuardrailSeverity.WARNING,
                guardrail_key=self.registry_key,
                message=(
                    f"Could not verify {len(errored)} of {len(pairs)} code(s) "
                    "-- the terminology backend is unavailable, unlicensed, or "
                    "does not cover them (a partial vocabulary cannot say a "
                    "code is invalid)."
                ),
                details={"unverifiable": errored},
            )
        return GuardrailResult(
            passed=True,
            severity=GuardrailSeverity.INFO,
            guardrail_key=self.registry_key,
            message=f"All {len(pairs)} code(s) valid.",
        )
