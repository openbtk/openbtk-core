"""Groundedness/faithfulness scoring: hand-computed claim counts against the
real GroundednessGuardrail (default heuristic and an injected checker)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from openbtk.core.errors import ProcessingError
from openbtk.core.schemas import GuardrailResult, GuardrailSeverity
from openbtk.eval.groundedness import (
    DetectorReport,
    GroundedExample,
    LabelledExample,
    evaluate_detector,
    groundedness_manifest,
    score_groundedness,
)
from openbtk.guardrails.groundedness import (
    GroundednessCheckInput,
    GroundednessGuardrail,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

_CONTEXT = ["Assessment: type 2 diabetes mellitus, stable. Metformin continued."]


def _ex(example_id: str, answer: str) -> GroundedExample:
    return GroundedExample(example_id=example_id, answer=answer, context=_CONTEXT)


class TestGuardrailClaimCount:
    """score_groundedness relies on the guardrail always reporting claim_count."""

    @pytest.mark.parametrize(
        ("answer", "count"),
        [
            ("The patient has type 2 diabetes.", 1),
            ("The patient has a fractured femur.", 1),
            ("Type 2 diabetes is stable. Metformin continued.", 2),
            ("", 0),
        ],
    )
    def test_claim_count_is_present_on_pass_and_fail(
        self, answer: str, count: int
    ) -> None:
        result = GroundednessGuardrail().check(
            GroundednessCheckInput(answer=answer, context=_CONTEXT)
        )
        assert result.details["claim_count"] == count


class TestScoreGroundedness:
    def test_counts_worked_out_by_hand(self) -> None:
        examples = [
            # 2 claims, both overlap the context strongly -> grounded
            _ex("a", "Type 2 diabetes is stable. Metformin continued."),
            # 1 claim, no overlap -> ungrounded
            _ex("b", "The patient has a fractured femur."),
            # 2 claims: first supported, second not
            _ex("c", "Type 2 diabetes mellitus. Prescribed warfarin anticoagulation."),
        ]
        report = score_groundedness(examples)
        assert report.n_examples == 3
        assert report.n_claims == 5
        assert report.unsupported_claims == 2
        assert report.supported_claims == 3
        assert report.faithfulness == pytest.approx(3 / 5)
        assert report.fully_grounded_examples == 1
        assert report.grounded_rate == pytest.approx(1 / 3)

    def test_an_empty_run_is_defined(self) -> None:
        report = score_groundedness([])
        assert (report.n_examples, report.n_claims) == (0, 0)
        assert report.faithfulness == 1.0 and report.grounded_rate == 1.0

    def test_an_injected_stricter_checker_changes_the_score(self) -> None:
        """The metric is only as good as the checker: swap in one that supports
        nothing and faithfulness collapses to 0 on the same input."""
        strict = GroundednessGuardrail(is_supported=lambda claim, context: False)
        report = score_groundedness([_ex("a", "Type 2 diabetes is stable.")], strict)
        assert report.faithfulness == 0.0

    def test_a_checker_that_does_not_report_claim_counts_is_refused(self) -> None:
        class Silent(GroundednessGuardrail):
            def check(self, payload: object) -> GuardrailResult:
                return GuardrailResult(
                    passed=True,
                    severity=GuardrailSeverity.INFO,
                    guardrail_key="x",
                    message="ok",
                )

        with pytest.raises(ProcessingError, match="claim_count") as exc:
            score_groundedness([_ex("a", "Type 2 diabetes.")], Silent())
        assert exc.value.context == {"example_id": "a"}

    def test_examples_stream(self) -> None:
        def stream() -> Iterator[GroundedExample]:
            for i in range(500):
                yield _ex(str(i), "Type 2 diabetes is stable.")

        assert score_groundedness(stream()).n_examples == 500

    def test_the_report_holds_counts_never_text(self) -> None:
        sentinel = "SECRETANSWERTEXT"
        report = score_groundedness([_ex("a", sentinel + " is unrelated.")])
        assert sentinel not in report.model_dump_json()
        assert sentinel not in str(report.as_dict())


class TestEvaluateDetector:
    def _labelled(
        self, example_id: str, answer: str, grounded: bool
    ) -> LabelledExample:
        return LabelledExample(
            example_id=example_id, answer=answer, context=_CONTEXT, grounded=grounded
        )

    def test_confusion_matrix_by_hand(self) -> None:
        examples = [
            self._labelled(
                "tp", "The patient has a fractured femur.", False
            ),  # flagged
            self._labelled("tn", "Type 2 diabetes is stable.", True),  # passed
            # Human says ungrounded (wrong dose) but the words all overlap: the
            # heuristic cannot see that -> a false negative.
            self._labelled("fn", "Metformin continued. Diabetes stable.", False),
            # Human says grounded (a paraphrase) but no words overlap -> flagged.
            self._labelled("fp", "Glycaemia is controlled.", True),
        ]
        report = evaluate_detector(examples)
        assert (
            report.true_positives,
            report.false_positives,
            report.false_negatives,
            report.true_negatives,
        ) == (1, 1, 1, 1)
        assert report.precision == 0.5 and report.recall == 0.5
        assert report.f1 == 0.5

    def test_undefined_ratios_are_one_and_zero_tp_f1_is_zero(self) -> None:
        empty = DetectorReport(
            true_positives=0, false_positives=0, false_negatives=0, true_negatives=0
        )
        assert (empty.precision, empty.recall) == (1.0, 1.0)
        miss = DetectorReport(
            true_positives=0, false_positives=1, false_negatives=1, true_negatives=0
        )
        assert miss.f1 == 0.0

    def test_as_dict_includes_derived_scores(self) -> None:
        report = DetectorReport(
            true_positives=3, false_positives=1, false_negatives=1, true_negatives=5
        )
        d = report.as_dict()
        assert d["precision"] == 0.75 and d["recall"] == 0.75 and d["f1"] == 0.75


class TestManifest:
    def test_records_kind_time_and_counts_but_no_text(self) -> None:
        sentinel = "SECRETANSWERTEXT"
        report = score_groundedness([_ex("a", sentinel + " and more.")])
        manifest = groundedness_manifest(report)
        assert manifest.kind == "groundedness"
        assert manifest.report["n_claims"] == report.n_claims
        assert manifest.started_at == report.started_at
        assert sentinel not in manifest.model_dump_json()
