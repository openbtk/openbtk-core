"""De-identification -- the flagship subsystem.

Cross-modal by design: free-text PHI removal, DICOM tag scrubbing and FHIR
field redaction are the same problem on different surfaces. See ADR-0006.

Public API (docs/04_API_DESIGN.md section 5):

    from openbtk.deid import DeidEngine, DeidMode

    engine = DeidEngine(mode=DeidMode.SURROGATE, recognizers=["rule"])
    result = engine.deidentify(text, patient_id="hashed-123")

``DeidEngine`` has zero heavy dependencies today (its one built-in
recognizer, ``RuleRecognizer``, is stdlib ``re`` only), so importing this
package eagerly is safe -- unlike ``openbtk.deid.recognizers``, whose
future NER/LLM-verifier recognizers must stay lazily imported.
"""

from __future__ import annotations

from openbtk.deid.engine import DeidEngine
from openbtk.deid.schemas import (
    DeidMode,
    DeidReport,
    DeidResult,
    DeidStatus,
    Detection,
    PHICategory,
    RiskEstimate,
)

__all__ = [
    "DeidEngine",
    "DeidMode",
    "DeidReport",
    "DeidResult",
    "DeidStatus",
    "Detection",
    "PHICategory",
    "RiskEstimate",
]
