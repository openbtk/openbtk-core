"""``DeidEngine``: orchestrates recognizers, merging, and a transform into
one call (docs/04_API_DESIGN.md section 5, ADR-0006).

    from openbtk.deid import DeidEngine, DeidMode

    engine = DeidEngine(mode=DeidMode.SURROGATE, recognizers=["rule"])
    result = engine.deidentify(text, patient_id="hashed-123")
    result.text      # de-identified text
    result.report    # DeidReport

**``recall_bias`` is an undocumented-beyond-its-type-signature knob**
(docs/04_API_DESIGN.md section 5 names the three levels; nothing anywhere
specifies the actual thresholds). The mapping below
(``_RECALL_BIAS_THRESHOLDS``) is a documented, reasonable-sounding default,
**not a benchmarked or tuned value** -- CLAUDE.md rule 14 forbids treating
it as one. Recalibrating it against real per-category F1 data is exactly
what ``tests/accuracy/`` (task 2.10) and a future i2b2/n2c2 harness are for.

**``residual_risk`` is a coarse, explainable heuristic**, not a calibrated
risk model: bucketed by the lowest confidence among surviving detections
(after recall-bias filtering), with an explicit, honest rationale for the
zero-detections case -- "found nothing" is never reported as proof of
"contains nothing".
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import TYPE_CHECKING, Literal

from openbtk.core.errors import DeidError
from openbtk.deid.merger import SpanMerger
from openbtk.deid.recognizers.base import RECOGNIZER_REGISTRY
from openbtk.deid.schemas import (
    DeidMode,
    DeidReport,
    DeidResult,
    PHICategory,
    RiskEstimate,
)
from openbtk.deid.transforms import Transform

if TYPE_CHECKING:
    from collections.abc import Sequence

    from openbtk.deid.consistency import ConsistencyStore
    from openbtk.deid.recognizers.base import BaseRecognizer
    from openbtk.deid.schemas import Detection

RecallBias = Literal["high", "balanced", "precision"]

_RECALL_BIAS_THRESHOLDS: dict[RecallBias, float] = {
    "high": 0.0,
    "balanced": 0.5,
    "precision": 0.8,
}
"""Minimum post-merge confidence a detection must clear to survive. See
this module's docstring: a documented default, not a tuned one."""

_DEFAULT_RECOGNIZERS: tuple[str, ...] = ("rule",)
"""ADR-0006's example config includes "ner"; only "rule" exists as of M2 --
requesting "ner" or "llm_verifier" today raises RegistryError naming the
missing key, which is the honest failure mode until tasks 2.4/2.9 land."""


class DeidEngine:
    """See module docstring."""

    def __init__(
        self,
        *,
        mode: DeidMode = DeidMode.SURROGATE,
        recall_bias: RecallBias = "high",
        recognizers: Sequence[str] = _DEFAULT_RECOGNIZERS,
        consistency_key: bytes | None = None,
        consistency_store: ConsistencyStore | None = None,
    ) -> None:
        if recall_bias not in _RECALL_BIAS_THRESHOLDS:
            raise DeidError(
                f"Unknown recall_bias {recall_bias!r}.",
                context={
                    "recall_bias": recall_bias,
                    "valid": list(_RECALL_BIAS_THRESHOLDS),
                },
            )
        self._mode = mode
        self._recall_bias = recall_bias
        self._recognizer_names = list(recognizers)
        self._recognizers = [self._resolve_recognizer(name) for name in recognizers]
        self._merger = SpanMerger()
        self._transform = Transform(
            mode, key=consistency_key, consistency_store=consistency_store
        )
        self._config_hash = self._compute_config_hash()

    @staticmethod
    def _resolve_recognizer(name: str) -> BaseRecognizer:
        """Recognizer short names (docs/04_API_DESIGN.md section 5's
        `recognizers=["rule", "ner"]`) map to `recognizer.general.<name>` --
        every recognizer registered so far uses the "general" scope."""
        key = f"recognizer.general.{name}"
        return RECOGNIZER_REGISTRY.create(key)

    def _compute_config_hash(self) -> str:
        # Deliberately NOT the consistency key -- this hash identifies the
        # engine's OPERATIONAL configuration for a run manifest / report,
        # not a secret. Plain SHA-256 is fine here, unlike a PHI-derived
        # value: there is nothing here to dictionary-attack.
        canonical = json.dumps(
            {
                "mode": self._mode.value,
                "recall_bias": self._recall_bias,
                "recognizers": self._recognizer_names,
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def deidentify(
        self, text: str, *, patient_id: str, document_id: str | None = None
    ) -> DeidResult:
        """Detect, merge, filter, transform, and report -- one document.

        Args:
            text: The document text to de-identify.
            patient_id: Stable identifier used for SURROGATE/DATE_SHIFT
                consistency. Should already be a hashed/pseudonymous value
                by the time it reaches here -- this engine has no way to
                enforce that.
            document_id: Report identifier. Auto-generated (a random UUID)
                if not given -- never derived from the text itself, which
                would risk encoding content into an identifier.

        Returns:
            The de-identified text plus its ``DeidReport``.
        """
        doc_id = document_id if document_id is not None else uuid.uuid4().hex
        raw_detections: list[Detection] = []
        for recognizer in self._recognizers:
            raw_detections.extend(recognizer.detect(text))

        merged = self._merger.merge(raw_detections)
        threshold = _RECALL_BIAS_THRESHOLDS[self._recall_bias]
        surviving = [d for d in merged if d.confidence >= threshold]

        deidentified_text = self._transform.apply(
            text, surviving, patient_id=patient_id
        )

        entity_counts: dict[PHICategory, int] = {}
        for detection in surviving:
            entity_counts[detection.category] = (
                entity_counts.get(detection.category, 0) + 1
            )

        report = DeidReport(
            document_id=doc_id,
            entity_counts=entity_counts,
            detections=surviving,
            residual_risk=self._estimate_risk(surviving),
            engine_config_hash=self._config_hash,
        )
        return DeidResult(text=deidentified_text, report=report)

    @staticmethod
    def _estimate_risk(detections: Sequence[Detection]) -> RiskEstimate:
        if not detections:
            return RiskEstimate(
                level="medium",
                rationale=(
                    "No PHI detected by the active recognizer ensemble. "
                    "This does not guarantee the document contains none -- "
                    "only that nothing surviving recognizers found was "
                    "reported as low confidence risk, in line with a "
                    "recall-biased default."
                ),
            )
        lowest = min(d.confidence for d in detections)
        if lowest < 0.5:
            level: Literal["low", "medium", "high"] = "high"
        elif lowest < 0.8:
            level = "medium"
        else:
            level = "low"
        return RiskEstimate(
            level=level,
            rationale=(
                f"{len(detections)} detection(s) survived filtering; "
                f"lowest post-merge confidence was {lowest:.2f}."
            ),
        )
