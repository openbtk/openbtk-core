"""De-identification benchmark scoring (roadmap task 8.1).

This is the harness M2's accuracy gate (tests/accuracy/) and the credentialed
n2c2/i2b2 benchmark both run: one scoring implementation, so a number
published from one and a regression gate on the other can never disagree
about what was measured.

**Matching rule, stated plainly because it is a real simplification, not the
industry-standard strict boundary match.** A ground-truth span counts as
found if ANY predicted detection overlaps it by at least one character. This
is entity-level "relaxed" matching. Two views are scored in one pass:

* **category-aware** (``per_category`` / ``overall``): the detection must also
  carry the SAME ``PHICategory``. Precision, recall and F1 per Safe Harbor
  category.
* **binary** (``binary``): category ignored -- was the span detected as *any*
  PHI? This is the leak-relevant view (a name caught but mislabelled as an
  organisation is still removed) and mirrors the i2b2 de-identification
  shared task's own binary PHI/non-PHI framing, though not its token-level
  scoring.

A detection is a false positive if it overlaps no ground-truth span of its
kind (category-aware) or of any kind (binary). Overlap is over the half-open
ranges ``[start, end)``.

**Nothing here can carry document text.** Inputs expose text to the engine
under test, but results hold only counts and category names -- a report from a
credentialed dataset must be safe to share.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field

from openbtk.deid.schemas import PHICategory  # noqa: TC001 -- Pydantic field type

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from openbtk.deid.engine import DeidEngine
    from openbtk.deid.schemas import Detection


class _SpanLike(Protocol):
    @property
    def category(self) -> PHICategory: ...
    @property
    def start(self) -> int: ...
    @property
    def end(self) -> int: ...


class _DocumentLike(Protocol):
    @property
    def document_id(self) -> str: ...
    @property
    def text(self) -> str: ...
    @property
    def spans(self) -> Sequence[_SpanLike]: ...


class CategoryMetrics(BaseModel):
    """True/false positive and false-negative counts, with derived scores.

    An undefined ratio (no predictions, or no ground truth) is reported as
    1.0 -- "nothing to get wrong" -- matching M2's accuracy gate; consult
    ``true_positives``/``false_negatives`` to tell "perfect" from "vacuous".

    Example:
        >>> m = CategoryMetrics(true_positives=3, false_positives=1, false_negatives=1)
        >>> m.precision, m.recall
        (0.75, 0.75)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    true_positives: int = Field(..., ge=0)
    false_positives: int = Field(..., ge=0)
    false_negatives: int = Field(..., ge=0)

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


class EvaluationResult(BaseModel):
    """Scores for one benchmark run. Counts and category names only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    per_category: dict[PHICategory, CategoryMetrics] = Field(default_factory=dict)
    binary: CategoryMetrics = Field(
        default_factory=lambda: CategoryMetrics(
            true_positives=0, false_positives=0, false_negatives=0
        ),
        description="Category-agnostic: was each span detected as any PHI?",
    )
    n_documents: int = Field(0, ge=0)
    unscored_gold_spans: int = Field(
        0,
        ge=0,
        description=(
            "Ground-truth annotations with no Safe Harbor category (e.g. "
            "i2b2 PROFESSION/AGE) that were not scored."
        ),
    )

    @property
    def overall(self) -> CategoryMetrics:
        """Micro-averaged over categories (category-aware)."""
        return CategoryMetrics(
            true_positives=sum(m.true_positives for m in self.per_category.values()),
            false_positives=sum(m.false_positives for m in self.per_category.values()),
            false_negatives=sum(m.false_negatives for m in self.per_category.values()),
        )


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def _score_document(
    doc: _DocumentLike, detections: Sequence[Detection]
) -> tuple[dict[PHICategory, list[int]], list[int]]:
    """Returns ``({category: [tp, fp, fn]}, [binary_tp, binary_fp, binary_fn])``."""
    counts: dict[PHICategory, list[int]] = {}
    binary = [0, 0, 0]

    def bump(category: PHICategory, idx: int) -> None:
        counts.setdefault(category, [0, 0, 0])[idx] += 1

    category_matched: set[int] = set()
    any_matched: set[int] = set()
    for gold in doc.spans:
        found_category = False
        found_any = False
        for i, det in enumerate(detections):
            if not _overlaps(gold.start, gold.end, det.span.start, det.span.end):
                continue
            found_any = True
            any_matched.add(i)
            if det.category == gold.category:
                found_category = True
                category_matched.add(i)
        bump(gold.category, 0 if found_category else 2)
        binary[0 if found_any else 2] += 1

    for i, det in enumerate(detections):
        if i not in category_matched:
            bump(det.category, 1)
        if i not in any_matched:
            binary[1] += 1
    return counts, binary


def evaluate_deid(
    engine: DeidEngine,
    documents: Iterable[_DocumentLike],
) -> EvaluationResult:
    """Run ``engine`` over ``documents`` and score it against their
    ground-truth spans.

    ``documents`` is consumed once (streaming, ADR-0004): a full n2c2 corpus
    is never held in memory. A document's optional ``unscored_spans`` count
    (see ``openbtk.deid.labelled.LabelledDocument``) is summed into the
    result so a report states how much annotation it did not score.
    """
    totals: dict[PHICategory, list[int]] = {}
    binary_totals = [0, 0, 0]
    unscored = 0
    n = 0
    for doc in documents:
        result = engine.deidentify(doc.text, patient_id=doc.document_id)
        counts, binary = _score_document(doc, list(result.report.detections))
        for category, (tp, fp, fn) in counts.items():
            bucket = totals.setdefault(category, [0, 0, 0])
            bucket[0] += tp
            bucket[1] += fp
            bucket[2] += fn
        for j in range(3):
            binary_totals[j] += binary[j]
        unscored += int(getattr(doc, "unscored_spans", 0))
        n += 1
    return EvaluationResult(
        per_category={
            category: CategoryMetrics(
                true_positives=v[0], false_positives=v[1], false_negatives=v[2]
            )
            for category, v in totals.items()
        },
        binary=CategoryMetrics(
            true_positives=binary_totals[0],
            false_positives=binary_totals[1],
            false_negatives=binary_totals[2],
        ),
        n_documents=n,
        unscored_gold_spans=unscored,
    )


def report_dict(result: EvaluationResult) -> dict[str, object]:
    """A JSON-serialisable report including the derived precision/recall/F1
    (Pydantic properties are not part of ``model_dump``). Counts and category
    names only -- safe to share.
    """

    def scores(m: CategoryMetrics) -> dict[str, float | int]:
        return {
            "true_positives": m.true_positives,
            "false_positives": m.false_positives,
            "false_negatives": m.false_negatives,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
        }

    return {
        "n_documents": result.n_documents,
        "unscored_gold_spans": result.unscored_gold_spans,
        "overall": scores(result.overall),
        "binary": scores(result.binary),
        "per_category": {
            c.value: scores(m)
            for c, m in sorted(result.per_category.items(), key=lambda kv: kv[0].value)
        },
    }


def format_markdown(
    result: EvaluationResult, *, title: str = "De-identification"
) -> str:
    """Render ``result`` as a Markdown report: counts and scores only, never
    any document text."""
    lines = [
        f"### {title}",
        "",
        f"{result.n_documents} documents; matching: relaxed span overlap "
        "(any overlapping detection counts).",
        "",
        "| Category | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for category in sorted(result.per_category, key=lambda c: c.value):
        m = result.per_category[category]
        lines.append(_row(category.value, m))
    lines.append(_row("**overall (category-aware)**", result.overall))
    lines.append(_row("**binary (any PHI)**", result.binary))
    if result.unscored_gold_spans:
        lines += [
            "",
            f"{result.unscored_gold_spans} ground-truth annotations had no Safe "
            "Harbor category and were not scored.",
        ]
    return "\n".join(lines) + "\n"


def _row(label: str, m: CategoryMetrics) -> str:
    return (
        f"| {label} | {m.true_positives} | {m.false_positives} | "
        f"{m.false_negatives} | {m.precision:.3f} | {m.recall:.3f} | {m.f1:.3f} |"
    )
