"""De-identification evaluation for the accuracy gate.

The scoring implementation lives in ``openbtk.eval.deid`` (M8 task 8.1) so
this regression gate and the credentialed n2c2/i2b2 benchmark run *the same
code* -- a published number and a CI gate can never disagree about what was
measured. This module only adapts the fixture corpus (which wraps its
documents in ``LabelledPHICorpus``) to that streaming interface; the matching
rule is documented once, in ``openbtk.eval.deid``'s module docstring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from openbtk.eval.deid import CategoryMetrics, EvaluationResult
from openbtk.eval.deid import evaluate_deid as _evaluate_deid

if TYPE_CHECKING:
    from fixtures.labelled_phi_corpus import LabelledPHICorpus
    from openbtk.deid.engine import DeidEngine

__all__ = ["CategoryMetrics", "EvaluationResult", "evaluate_deid"]


def evaluate_deid(engine: DeidEngine, corpus: LabelledPHICorpus) -> EvaluationResult:
    """Run ``engine`` over every document in ``corpus``."""
    return _evaluate_deid(engine, corpus.documents)
