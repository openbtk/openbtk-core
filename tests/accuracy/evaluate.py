"""Span-overlap-based de-identification evaluation against a
LabelledPHICorpus.

docs/07_TEST_CHARTER.md section 3.6: "De-identification F1 against the
bundled synthetic labelled corpus, gated in CI." "Per-category recall is
asserted separately, so an improvement in name detection cannot mask a
regression in MRN detection."

**Matching rule, stated plainly because it is a real simplification, not
an industry-standard exact-boundary match**: a ground-truth span counts as
found if ANY predicted detection of the SAME category overlaps it by at
least one character. This is a document-level presence check per span, not
a strict IOU/boundary-exact NER metric -- appropriate for what this
milestone's engine actually needs to prove (did the PHI get caught at
all?), not boundary precision down to the character.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fixtures.labelled_phi_corpus import LabelledDocument, LabelledPHICorpus
    from openbtk.deid.engine import DeidEngine
    from openbtk.deid.schemas import Detection, PHICategory


@dataclass(frozen=True)
class CategoryMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass(frozen=True)
class EvaluationResult:
    per_category: dict[PHICategory, CategoryMetrics] = field(default_factory=dict)

    @property
    def overall(self) -> CategoryMetrics:
        tp = sum(m.true_positives for m in self.per_category.values())
        fp = sum(m.false_positives for m in self.per_category.values())
        fn = sum(m.false_negatives for m in self.per_category.values())
        return CategoryMetrics(
            true_positives=tp, false_positives=fp, false_negatives=fn
        )


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def _evaluate_document(
    doc: LabelledDocument, detections: list[Detection]
) -> dict[PHICategory, tuple[int, int, int]]:
    """Returns {category: (tp, fp, fn)} for one document."""
    counts: dict[PHICategory, list[int]] = {}

    def bump(category: PHICategory, idx: int) -> None:
        counts.setdefault(category, [0, 0, 0])[idx] += 1

    matched_detection_indices: set[int] = set()
    for gt_span in doc.spans:
        found = False
        for i, det in enumerate(detections):
            if det.category != gt_span.category:
                continue
            if _overlaps(gt_span.start, gt_span.end, det.span.start, det.span.end):
                found = True
                matched_detection_indices.add(i)
        bump(gt_span.category, 0 if found else 2)  # 0=tp, 2=fn

    for i, det in enumerate(detections):
        if i not in matched_detection_indices:
            bump(det.category, 1)  # fp

    return {cat: (v[0], v[1], v[2]) for cat, v in counts.items()}


def evaluate_deid(engine: DeidEngine, corpus: LabelledPHICorpus) -> EvaluationResult:
    """Run ``engine`` over every document in ``corpus`` and compute
    per-category precision/recall/F1 against its ground-truth spans."""
    totals: dict[PHICategory, list[int]] = {}
    for doc in corpus.documents:
        result = engine.deidentify(doc.text, patient_id=doc.document_id)
        per_doc = _evaluate_document(doc, list(result.report.detections))
        for category, (tp, fp, fn) in per_doc.items():
            bucket = totals.setdefault(category, [0, 0, 0])
            bucket[0] += tp
            bucket[1] += fp
            bucket[2] += fn

    per_category = {
        category: CategoryMetrics(
            true_positives=v[0], false_positives=v[1], false_negatives=v[2]
        )
        for category, v in totals.items()
    }
    return EvaluationResult(per_category=per_category)
