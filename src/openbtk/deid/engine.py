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

import contextlib
import hashlib
import importlib
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
"""ADR-0006's example config includes "ner" too; requesting it opts in
explicitly rather than being on by default, since it needs a downloaded
spaCy model (openbtk.deid.recognizers.ner's own docstring) that a bare
`pip install openbtk[text]` doesn't fetch automatically. "llm_verifier"
(task 2.9) doesn't exist yet -- requesting it raises RegistryError naming
the missing key, the honest failure mode until it's built."""


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
        # DeidMode(mode) is idempotent for an already-real DeidMode member
        # (enum construction from an existing member returns that member),
        # and coerces a plain string -- which is exactly what every
        # registry/config-driven construction supplies (StepConfig.params
        # is JSON-safe only, never a live enum member; docs/03_ARCHITECTURE.md
        # section 7.3's own worked YAML example passes `mode: surrogate` as
        # a bare string). Without this, Transform's `self._mode is
        # DeidMode.REDACT`-style identity checks silently never match a
        # plain string, and _compute_config_hash's `self._mode.value`
        # crashes outright -- both confirmed by direct reproduction, not
        # assumed.
        try:
            mode = DeidMode(mode)
        except ValueError as e:
            raise DeidError(
                f"Unknown mode {mode!r}.",
                context={"mode": str(mode), "valid": [m.value for m in DeidMode]},
            ) from e
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
        every recognizer registered so far uses the "general" scope.

        A recognizer module beyond `rule` (e.g. `ner`) is never imported by
        `openbtk.deid.recognizers.__init__` -- that would force every
        caller of `openbtk.deid`, including the default test suite, to pay
        for spaCy and a downloaded model whether or not "ner" is ever
        requested. So resolving a name not yet registered tries importing
        `openbtk.deid.recognizers.<name>` first, as the one place that
        opt-in actually happens. A name that isn't a real module (e.g. the
        not-yet-built "llm_verifier") falls through to RECOGNIZER_REGISTRY's
        own RegistryError, unchanged.
        """
        key = f"recognizer.general.{name}"
        if not RECOGNIZER_REGISTRY.is_registered(key):
            # let RECOGNIZER_REGISTRY.create() raise its own RegistryError
            # for a name that isn't a real module either
            with contextlib.suppress(ModuleNotFoundError):
                importlib.import_module(f"openbtk.deid.recognizers.{name}")
        return RECOGNIZER_REGISTRY.create(key)

    @staticmethod
    def _shield_rule_detections_from_ner(
        detections: list[Detection],
    ) -> list[Detection]:
        """Drop any "ner" detection that overlaps a "rule" detection.

        ADR-0006 names this exact mitigation ("high-precision rules run
        first and their spans are excluded from NER re-examination where
        safe") for performance; it turns out to matter just as much for
        correctness. Measured directly against the labelled corpus: a
        general-purpose NER model over structured "Label: VALUE" text
        (license numbers, URLs, even a field label like "License:" alone)
        produces wide, wrong PERSON spans that -- left unfiltered --
        overlap and, per SpanMerger's documented "widest span wins" policy,
        WIN OVER a correct, narrow, high-confidence rule detection. Without
        this shield, adding NER measurably regressed already-perfect
        categories (certificate_license_number, url) instead of only
        adding NAME/GEOGRAPHIC_SUBDIVISION coverage -- caught by
        tests/accuracy/, not assumed away.

        Rule detections are never dropped by this filter, only NER ones;
        two NER detections overlapping each other, or a "rule" overlapping
        another "rule", still go to SpanMerger exactly as before.
        """
        rule_spans = [
            (d.span.start, d.span.end) for d in detections if d.method == "rule"
        ]
        if not rule_spans:
            return detections

        def _overlaps_a_rule_span(detection: Detection) -> bool:
            return any(
                detection.span.start < end and start < detection.span.end
                for start, end in rule_spans
            )

        return [
            d for d in detections if d.method != "ner" or not _overlaps_a_rule_span(d)
        ]

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
        raw_detections = self._shield_rule_detections_from_ner(raw_detections)

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
