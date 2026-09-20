"""Clinical guardrails: PHI leakage, terminology validity, groundedness,
dose plausibility, EHR-specific checks (docs/03_ARCHITECTURE.md section 8.3).

Cross-modal -- a guardrail is not owned by a modality module (FR-G-08).
``GuardrailPipeline`` (``openbtk.guardrails.pipeline``) composes several
guardrails over one payload; it is not itself registered (see its own
module docstring).

Submodules are imported here (registering every guardrail as a side
effect) for the same reason every other component package does: none of
them need a heavy optional dependency merely to be *defined*.
"""

from __future__ import annotations

from openbtk.guardrails import (
    dose,
    ehr,
    groundedness,
    phi_leakage,
    terminology_validity,
)
from openbtk.guardrails.pipeline import GuardrailPipeline, GuardrailPipelineResult

__all__ = [
    "GuardrailPipeline",
    "GuardrailPipelineResult",
    "dose",
    "ehr",
    "groundedness",
    "phi_leakage",
    "terminology_validity",
]
