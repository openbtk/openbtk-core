"""De-identification accuracy regression gate (docs/07_TEST_CHARTER.md
section 3.6, M2 task 2.10).

Runs ``DeidEngine(recognizers=["rule"])`` -- the only recognizer that
exists at M2 -- over the full ``labelled_phi_corpus`` and compares against
``baseline.json``, checked into the repo. "Per-category recall is asserted
separately, so an improvement in name detection cannot mask a regression
in MRN detection" (section 3.6): ``test_no_per_category_regression`` is
parametrized over every category in the baseline individually, not folded
into one aggregate check.

**Read this before "fixing" a failure by editing baseline.json.** Two of
the sixteen text-representable categories -- ``NAME`` and
``GEOGRAPHIC_SUBDIVISION`` -- are pinned at 0.0 recall in the baseline
*by design*, not as a target to leave alone forever: ``RuleRecognizer``
does not attempt them at all (see its own module docstring; that is
``NERRecognizer``'s job, ADR-0006 task 2.4, not yet built). If this file
starts failing because those two categories now score ABOVE 0.0, that is
good news -- update the baseline deliberately, with a PR that says why
(section 3.6's own rule), rather than treating it as a regression to
suppress.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openbtk.deid.engine import DeidEngine
from openbtk.deid.schemas import PHICategory

from .evaluate import EvaluationResult, evaluate_deid

_TOLERANCE = 0.005
"""Per docs/07_TEST_CHARTER.md section 3.6's own template: "tolerance for
nondeterminism." Rule-based detection over a fixed-seed corpus is actually
fully deterministic today -- this exists for forward compatibility with a
future nondeterministic recognizer (NER, an LLM verifier), not because
today's numbers wobble."""

_BASELINE_PATH = Path(__file__).parent / "baseline.json"


def _load_baseline() -> dict[str, object]:
    with _BASELINE_PATH.open(encoding="utf-8") as f:
        data: dict[str, object] = json.load(f)
    return data


def _evaluate_rule_only_engine() -> EvaluationResult:
    from fixtures.labelled_phi_corpus import build_labelled_phi_corpus

    corpus = build_labelled_phi_corpus()
    engine = DeidEngine(recognizers=["rule"])
    return evaluate_deid(engine, corpus)


@pytest.fixture(scope="module")
def evaluation_result() -> EvaluationResult:
    return _evaluate_rule_only_engine()


@pytest.fixture(scope="module")
def baseline() -> dict[str, object]:
    return _load_baseline()


class TestOverallF1:
    def test_f1_meets_the_checked_in_baseline(
        self, evaluation_result: EvaluationResult, baseline: dict[str, object]
    ) -> None:
        overall = baseline["overall"]
        assert isinstance(overall, dict)
        assert evaluation_result.overall.f1 >= overall["f1"] - _TOLERANCE

    def test_recall_meets_the_checked_in_baseline(
        self, evaluation_result: EvaluationResult, baseline: dict[str, object]
    ) -> None:
        overall = baseline["overall"]
        assert isinstance(overall, dict)
        assert evaluation_result.overall.recall >= overall["recall"] - _TOLERANCE

    def test_precision_meets_the_checked_in_baseline(
        self, evaluation_result: EvaluationResult, baseline: dict[str, object]
    ) -> None:
        overall = baseline["overall"]
        assert isinstance(overall, dict)
        assert evaluation_result.overall.precision >= overall["precision"] - _TOLERANCE


class TestPerCategoryRegression:
    @pytest.mark.parametrize("category", list(PHICategory))
    def test_no_per_category_regression(
        self,
        category: PHICategory,
        evaluation_result: EvaluationResult,
        baseline: dict[str, object],
    ) -> None:
        per_category = baseline["per_category"]
        assert isinstance(per_category, dict)
        if category.value not in per_category:
            pytest.skip(
                f"{category.value} is not text-representable "
                "(PHICategory's own docstring) -- absent from the baseline "
                "by design, not an oversight."
            )
        expected = per_category[category.value]
        actual = evaluation_result.per_category.get(category)
        actual_recall = actual.recall if actual is not None else 0.0
        assert actual_recall >= expected["recall"] - _TOLERANCE, (
            f"{category.value}: recall regressed from "
            f"{expected['recall']:.3f} to {actual_recall:.3f}"
        )


class TestKnownGapIsExplicit:
    """Not a regression gate -- a loud, deliberate marker that these two
    categories are still at zero, so a passing suite never quietly reads
    as "de-identification is complete." """

    @pytest.mark.parametrize(
        "category", [PHICategory.NAME, PHICategory.GEOGRAPHIC_SUBDIVISION]
    )
    def test_rule_only_recall_is_still_zero_for_ner_only_categories(
        self, category: PHICategory, evaluation_result: EvaluationResult
    ) -> None:
        metrics = evaluation_result.per_category.get(category)
        assert metrics is not None
        assert metrics.recall == 0.0, (
            f"{category.value} now has nonzero recall -- NERRecognizer "
            "(or something) must have started detecting it. Update "
            "baseline.json deliberately (with a PR stating why, per "
            "docs/07_TEST_CHARTER.md section 3.6) rather than leaving this "
            "test broken."
        )


class TestZeroFalsePositives:
    """RuleRecognizer's patterns are format-specific enough that they
    should never fire on the corpus's surrounding narrative text -- a
    concrete, checkable claim, not an assumption."""

    def test_precision_is_perfect_on_the_current_corpus(
        self, evaluation_result: EvaluationResult
    ) -> None:
        assert evaluation_result.overall.precision == 1.0
