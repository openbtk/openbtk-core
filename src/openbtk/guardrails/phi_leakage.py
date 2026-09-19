"""``PHILeakageGuardrail``: flags PHI in generated output (FR-G-02).

Wraps the real ``openbtk.deid.DeidEngine`` detection machinery rather than
reinventing PHI detection a second time (docs/09_CODING_STANDARDS.md
section 7, "wrap, don't reinvent" -- applied here to this project's own
flagship subsystem, not just a third-party library). This guardrail
never returns or uses ``DeidEngine``'s transformed text: its only job is
to report whether PHI is present and where, so a caller can decide
whether to regenerate or block a response -- silently rewriting a
guardrail's own input would defeat the point of checking it.

``GuardrailResult.message``/``.details`` never carry the detected text
itself, only category names and counts -- ``TextSpan`` (in ``.spans``)
carries only offsets, a label and a confidence, never a substring
(docs/06_SECURITY_COMPLIANCE.md: never log raw_text).
"""

from __future__ import annotations

from typing import Any

from openbtk.core.base import BaseGuardrail
from openbtk.core.registry import GUARDRAIL_REGISTRY
from openbtk.core.schemas import GuardrailResult, GuardrailSeverity
from openbtk.deid import DeidEngine, DeidMode

_GUARDRAIL_CHECK_PATIENT_ID = "guardrail-check"
"""A fixed, non-identifying placeholder. This guardrail discards
DeidEngine's transformed text entirely (only its detections matter), so
it never needs SURROGATE/DATE_SHIFT cross-call consistency -- no real
patient identifier is needed or appropriate here."""


@GUARDRAIL_REGISTRY.register("guardrail.general.phi_leakage")
class PHILeakageGuardrail(BaseGuardrail):
    """Flags PHI-shaped spans in a payload.

    Args:
        recognizers: Forwarded to the wrapped ``DeidEngine`` -- defaults
            to the same zero-extras-safe default ``DeidEngine`` itself
            uses (rule-based only; pass ``("rule", "ner")`` for the
            opt-in NER ensemble if the ``text`` extra is installed).

    Example:
        >>> import contextlib, io
        >>> with contextlib.redirect_stdout(io.StringIO()):  # registry logging
        ...     guardrail = PHILeakageGuardrail()
        ...     result = guardrail.check("Patient SSN: 123-45-6789.")
        ...     clean = guardrail.check("No identifiers here.")
        >>> result.passed
        False
        >>> result.severity.value
        'block'
        >>> clean.passed
        True
    """

    def __init__(self, *, recognizers: tuple[str, ...] = ("rule",)) -> None:
        self._engine = DeidEngine(mode=DeidMode.TAG, recognizers=list(recognizers))

    def check(self, payload: Any) -> GuardrailResult:
        text = _extract_text(payload)
        try:
            result = self._engine.deidentify(
                text, patient_id=_GUARDRAIL_CHECK_PATIENT_ID
            )
        except Exception as e:  # never raise -- see BaseGuardrail.check's contract
            return GuardrailResult(
                passed=False,
                severity=GuardrailSeverity.WARNING,
                guardrail_key=self.registry_key,
                message=f"PHI check itself failed: {type(e).__name__}.",
            )
        detections = result.report.detections
        if not detections:
            return GuardrailResult(
                passed=True,
                severity=GuardrailSeverity.INFO,
                guardrail_key=self.registry_key,
                message="No PHI detected.",
            )
        categories = sorted({d.category.value for d in detections})
        return GuardrailResult(
            passed=False,
            severity=GuardrailSeverity.BLOCK,
            guardrail_key=self.registry_key,
            message=(
                f"Detected {len(detections)} potential PHI span(s): "
                f"{', '.join(categories)}."
            ),
            spans=[d.span for d in detections],
            details={"categories": categories, "count": len(detections)},
        )


def _extract_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    text = getattr(payload, "text", None)
    if isinstance(text, str):
        return text
    return str(payload)
