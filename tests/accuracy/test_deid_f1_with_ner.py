"""De-identification accuracy for the FULL ensemble
(``recognizers=["rule", "ner"]``) -- the configuration
docs/04_API_DESIGN.md section 5's own example constructs explicitly.

Requires the real spaCy model (``pip install openbtk[text]`` and
``python -m spacy download en_core_web_sm``), so every test here is
``@pytest.mark.slow`` -- skipped by default, run with
``OPENBTK_SLOW_TESTS=1``. ``DeidEngine``'s zero-extras-safe *default*
(``recognizers=("rule",)``, tests/accuracy/test_deid_f1.py) is unaffected
by anything in this file.

**Read this before treating a change here as either a win or a
regression.** Adding NER is not a strict improvement over rule-only: it
trades some precision for a lot of recall on two previously-impossible
categories. The measured, checked-in numbers in ``baseline_with_ner.json``
are:

  * ``name``: recall 0.0 -> 0.96, precision drops to ~0.38. spaCy's
    general-purpose model mistakes structured "Label: VALUE" text --
    license numbers, URLs, even a bare field label like "License:" --
    for PERSON entities.
  * ``geographic_subdivision``: recall 0.0 -> only ~0.08. Faker's street
    addresses are not a shape ``en_core_web_sm`` reliably tags GPE/LOC/FAC.
  * Every category ``RuleRecognizer`` already covered stays at a perfect
    1.0 -- this depends on ``DeidEngine``'s rule/NER shield (ADR-0006's own
    named mitigation: "high-precision rules run first and their spans are
    excluded from NER re-examination"), verified directly by
    ``TestRuleCategoriesUnaffectedByNer`` below. Without that shield, wide
    false-positive NER spans measurably regressed already-perfect
    categories -- caught here, not assumed away.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openbtk.deid.engine import DeidEngine
from openbtk.deid.schemas import PHICategory

from .evaluate import EvaluationResult, evaluate_deid

pytestmark = pytest.mark.slow

_TOLERANCE = 0.005
_BASELINE_PATH = Path(__file__).parent / "baseline_with_ner.json"
_RULE_ONLY_CATEGORIES = {
    PHICategory.SSN,
    PHICategory.EMAIL,
    PHICategory.URL,
    PHICategory.IP_ADDRESS,
    PHICategory.DATE,
    PHICategory.PHONE_NUMBER,
    PHICategory.FAX_NUMBER,
    PHICategory.MEDICAL_RECORD_NUMBER,
    PHICategory.HEALTH_PLAN_BENEFICIARY_NUMBER,
    PHICategory.ACCOUNT_NUMBER,
    PHICategory.CERTIFICATE_LICENSE_NUMBER,
    PHICategory.VEHICLE_IDENTIFIER,
    PHICategory.DEVICE_IDENTIFIER,
    PHICategory.OTHER_UNIQUE_IDENTIFIER,
}


def _load_baseline() -> dict[str, object]:
    with _BASELINE_PATH.open(encoding="utf-8") as f:
        data: dict[str, object] = json.load(f)
    return data


@pytest.fixture(scope="module")
def evaluation_result() -> EvaluationResult:
    from fixtures.labelled_phi_corpus import build_labelled_phi_corpus

    corpus = build_labelled_phi_corpus()
    engine = DeidEngine(recognizers=["rule", "ner"])
    return evaluate_deid(engine, corpus)


@pytest.fixture(scope="module")
def baseline() -> dict[str, object]:
    return _load_baseline()


class TestOverallMetricsMatchBaseline:
    def test_f1(
        self, evaluation_result: EvaluationResult, baseline: dict[str, object]
    ) -> None:
        overall = baseline["overall"]
        assert isinstance(overall, dict)
        assert evaluation_result.overall.f1 >= overall["f1"] - _TOLERANCE

    def test_recall(
        self, evaluation_result: EvaluationResult, baseline: dict[str, object]
    ) -> None:
        overall = baseline["overall"]
        assert isinstance(overall, dict)
        assert evaluation_result.overall.recall >= overall["recall"] - _TOLERANCE


class TestRuleCategoriesUnaffectedByNer:
    """The regression this file exists to prevent from ever coming back
    silently: adding a noisy NER recognizer must never make an
    already-perfect rule-covered category worse."""

    @pytest.mark.parametrize("category", sorted(_RULE_ONLY_CATEGORIES, key=str))
    def test_rule_coverable_category_stays_at_perfect_recall(
        self, category: PHICategory, evaluation_result: EvaluationResult
    ) -> None:
        metrics = evaluation_result.per_category.get(category)
        assert metrics is not None
        assert metrics.recall == 1.0, (
            f"{category.value} recall dropped below 1.0 with NER enabled -- "
            "the rule/NER shield in DeidEngine may have regressed. See "
            "DeidEngine._shield_rule_detections_from_ner."
        )


class TestNerCategoriesShowRealImprovement:
    """Not "solved" -- measurably better than 0.0, which is the entire
    point of building NER at all. TestKnownGapIsExplicit in
    test_deid_f1.py already documents what rule-only achieves (zero); this
    is the honest other half of that comparison."""

    def test_name_recall_improves_substantially(
        self, evaluation_result: EvaluationResult
    ) -> None:
        metrics = evaluation_result.per_category.get(PHICategory.NAME)
        assert metrics is not None
        assert metrics.recall > 0.9

    def test_geographic_subdivision_recall_is_nonzero_but_still_weak(
        self, evaluation_result: EvaluationResult
    ) -> None:
        """Deliberately a narrow band, not a floor to celebrate: this
        documents a real, disclosed limitation (general-purpose NER does
        not reliably recognize street-address-shaped text), not a target."""
        metrics = evaluation_result.per_category.get(PHICategory.GEOGRAPHIC_SUBDIVISION)
        assert metrics is not None
        assert 0.0 < metrics.recall < 0.5
