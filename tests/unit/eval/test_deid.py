"""Unit tests for openbtk.eval.deid -- expected counts worked out by hand from
the ground truth and the (stubbed) detections, not read back from the code."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from openbtk.core.schemas import TextSpan
from openbtk.deid.labelled import LabelledDocument, LabelledSpan
from openbtk.deid.schemas import (
    DeidReport,
    DeidResult,
    Detection,
    PHICategory,
    RiskEstimate,
)
from openbtk.eval.deid import (
    CategoryMetrics,
    EvaluationResult,
    evaluate_deid,
    format_markdown,
    report_dict,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

_TEXT = "x" * 100


def _det(category: PHICategory, start: int, end: int) -> Detection:
    return Detection(
        category=category,
        span=TextSpan(start=start, end=end, label=category.value, confidence=0.9),
        confidence=0.9,
        method="rule",
    )


def _gold(category: PHICategory, start: int, end: int) -> LabelledSpan:
    return LabelledSpan(category=category, start=start, end=end)


class _StubEngine:
    """Returns pre-set detections per document id -- isolates the scorer from
    any recognizer's own behaviour."""

    def __init__(self, by_doc: dict[str, Sequence[Detection]]) -> None:
        self._by_doc = by_doc

    def deidentify(self, text: str, *, patient_id: str) -> DeidResult:
        return DeidResult(
            text=text,
            report=DeidReport(
                document_id=patient_id,
                entity_counts={},
                detections=list(self._by_doc.get(patient_id, [])),
                residual_risk=RiskEstimate(level="low", rationale="stub"),
                engine_config_hash="stub",
            ),
        )


def _run(
    docs: list[LabelledDocument], by_doc: dict[str, Sequence[Detection]]
) -> EvaluationResult:
    return evaluate_deid(_StubEngine(by_doc), docs)  # type: ignore[arg-type]


def _doc(doc_id: str, *spans: LabelledSpan, unscored: int = 0) -> LabelledDocument:
    return LabelledDocument(
        document_id=doc_id, text=_TEXT, spans=list(spans), unscored_spans=unscored
    )


class TestCategoryMetrics:
    def test_scores(self) -> None:
        m = CategoryMetrics(true_positives=6, false_positives=2, false_negatives=4)
        assert m.precision == 0.75
        assert m.recall == 0.6
        assert m.f1 == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6))

    def test_undefined_ratios_are_one_but_f1_of_zero_tp_is_zero(self) -> None:
        empty = CategoryMetrics(true_positives=0, false_positives=0, false_negatives=0)
        assert (empty.precision, empty.recall) == (1.0, 1.0)
        miss = CategoryMetrics(true_positives=0, false_positives=1, false_negatives=1)
        assert miss.f1 == 0.0


class TestMatching:
    def test_exact_detection_is_a_true_positive(self) -> None:
        result = _run(
            [_doc("d", _gold(PHICategory.DATE, 10, 20))],
            {"d": [_det(PHICategory.DATE, 10, 20)]},
        )
        assert result.per_category[PHICategory.DATE] == CategoryMetrics(
            true_positives=1, false_positives=0, false_negatives=0
        )

    def test_one_character_of_overlap_is_enough(self) -> None:
        result = _run(
            [_doc("d", _gold(PHICategory.DATE, 10, 20))],
            {"d": [_det(PHICategory.DATE, 19, 30)]},
        )
        assert result.per_category[PHICategory.DATE].true_positives == 1

    def test_adjacent_spans_do_not_overlap_half_open(self) -> None:
        result = _run(
            [_doc("d", _gold(PHICategory.DATE, 10, 20))],
            {"d": [_det(PHICategory.DATE, 20, 30)]},
        )
        m = result.per_category[PHICategory.DATE]
        assert (m.true_positives, m.false_positives, m.false_negatives) == (0, 1, 1)

    def test_a_missed_span_is_a_false_negative(self) -> None:
        result = _run([_doc("d", _gold(PHICategory.NAME, 0, 5))], {})
        assert result.per_category[PHICategory.NAME].false_negatives == 1

    def test_a_detection_over_nothing_is_a_false_positive(self) -> None:
        result = _run([_doc("d")], {"d": [_det(PHICategory.EMAIL, 0, 5)]})
        assert result.per_category[PHICategory.EMAIL].false_positives == 1

    def test_one_detection_covering_two_gold_spans_counts_both_found(self) -> None:
        result = _run(
            [
                _doc(
                    "d",
                    _gold(PHICategory.NAME, 0, 5),
                    _gold(PHICategory.NAME, 6, 10),
                )
            ],
            {"d": [_det(PHICategory.NAME, 0, 10)]},
        )
        m = result.per_category[PHICategory.NAME]
        assert (m.true_positives, m.false_positives, m.false_negatives) == (2, 0, 0)


class TestCategoryAwareVersusBinary:
    def test_a_mislabelled_detection_is_wrong_by_category_right_by_binary(
        self,
    ) -> None:
        """A name caught but called an email is a category miss (FN for name,
        FP for email) yet still removed -- the binary view counts it found."""
        result = _run(
            [_doc("d", _gold(PHICategory.NAME, 0, 8))],
            {"d": [_det(PHICategory.EMAIL, 0, 8)]},
        )
        name = result.per_category[PHICategory.NAME]
        email = result.per_category[PHICategory.EMAIL]
        assert (name.true_positives, name.false_negatives) == (0, 1)
        assert email.false_positives == 1
        assert (
            result.binary.true_positives,
            result.binary.false_positives,
            result.binary.false_negatives,
        ) == (1, 0, 0)

    def test_binary_false_positive_only_when_overlapping_nothing(self) -> None:
        result = _run(
            [_doc("d", _gold(PHICategory.NAME, 0, 8))],
            {"d": [_det(PHICategory.EMAIL, 0, 8), _det(PHICategory.DATE, 50, 60)]},
        )
        assert result.binary.false_positives == 1


class TestAggregation:
    def test_counts_sum_across_documents_and_overall_is_micro_averaged(self) -> None:
        docs = [
            _doc("a", _gold(PHICategory.DATE, 0, 10), _gold(PHICategory.NAME, 20, 30)),
            _doc("b"),
        ]
        by_doc = {
            "a": [_det(PHICategory.DATE, 0, 10)],
            "b": [_det(PHICategory.DATE, 40, 50)],
        }
        result = _run(docs, by_doc)
        # date: tp 1 (doc a), fp 1 (doc b);  name: fn 1.
        assert result.n_documents == 2
        assert result.overall == CategoryMetrics(
            true_positives=1, false_positives=1, false_negatives=1
        )
        assert result.overall.precision == 0.5
        assert result.overall.recall == 0.5

    def test_unscored_annotations_are_summed_from_the_documents(self) -> None:
        result = _run([_doc("a", unscored=2), _doc("b", unscored=3)], {})
        assert result.unscored_gold_spans == 5

    def test_documents_are_consumed_as_a_stream(self) -> None:
        def stream() -> Any:
            yield _doc("a", _gold(PHICategory.DATE, 0, 5))
            yield _doc("b")

        result = evaluate_deid(_StubEngine({}), stream())  # type: ignore[arg-type]
        assert result.n_documents == 2

    def test_empty_input_gives_an_empty_result(self) -> None:
        result = evaluate_deid(_StubEngine({}), [])  # type: ignore[arg-type]
        assert result.n_documents == 0
        assert result.overall.true_positives == 0


class TestReports:
    def _result(self) -> EvaluationResult:
        return _run(
            [_doc("a", _gold(PHICategory.DATE, 0, 10), unscored=1)],
            {"a": [_det(PHICategory.DATE, 0, 10)]},
        )

    def test_report_dict_includes_derived_scores(self) -> None:
        report = report_dict(self._result())
        assert report["n_documents"] == 1
        assert report["unscored_gold_spans"] == 1
        per_category = report["per_category"]
        assert isinstance(per_category, dict)
        assert per_category["date"]["f1"] == 1.0
        overall = report["overall"]
        assert isinstance(overall, dict)
        assert overall["precision"] == 1.0

    def test_markdown_lists_categories_and_the_unscored_note(self) -> None:
        md = format_markdown(self._result(), title="T")
        assert md.startswith("### T")
        assert "| date | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 |" in md
        assert "overall (category-aware)" in md
        assert "binary (any PHI)" in md
        assert "1 ground-truth annotations had no Safe Harbor category" in md

    def test_markdown_omits_the_unscored_note_when_zero(self) -> None:
        result = _run([_doc("a", _gold(PHICategory.DATE, 0, 5))], {})
        assert "no Safe Harbor category" not in format_markdown(result)

    def test_reports_never_contain_document_text(self) -> None:
        secret = "SECRETDOCUMENTTEXT"  # pragma: allowlist secret
        doc = LabelledDocument(
            document_id="d",
            text=secret,
            spans=[_gold(PHICategory.NAME, 0, 6)],
        )
        result = evaluate_deid(_StubEngine({}), [doc])  # type: ignore[arg-type]
        assert secret not in format_markdown(result)
        assert secret not in str(report_dict(result))
        assert secret not in result.model_dump_json()
