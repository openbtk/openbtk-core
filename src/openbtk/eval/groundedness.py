"""Groundedness / faithfulness scoring for generated clinical text (FR-X-04).

Two different questions, kept apart because conflating them is how a metric
gets over-trusted:

1. :func:`score_groundedness` -- *how grounded is this model's output?* Runs a
   :class:`~openbtk.guardrails.groundedness.GroundednessGuardrail` over
   ``(answer, context)`` pairs and reports the fraction of claims supported by
   their context (**faithfulness**), and how many answers were fully grounded.
2. :func:`evaluate_detector` -- *how good is the checker itself?* Given
   examples a human has labelled grounded or not, reports how well the
   guardrail's verdict agrees. Use this on your own labelled data before
   trusting question 1's number.

**The default checker is a word-overlap heuristic, not entailment** (see
``GroundednessGuardrail``). A faithfulness score from it measures lexical
overlap with the context; it cannot see a negation, a swapped dose, or a
paraphrase. Pass a guardrail built with your own ``is_supported`` (an NLI model
or an LLM judge) for a stronger check. This module publishes no faithfulness
figure for any model (CLAUDE.md rule 14).

Reports hold counts only -- never answer or context text.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from openbtk.core.errors import ProcessingError
from openbtk.eval.manifest import EvalManifest, build_manifest
from openbtk.guardrails.groundedness import (
    GroundednessCheckInput,
    GroundednessGuardrail,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from openbtk.core.schemas import JsonValue


class GroundedExample(BaseModel):
    """An answer and the source text it should be grounded in."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    example_id: str = Field(..., min_length=1)
    answer: str
    context: list[str] = Field(default_factory=list)


class LabelledExample(GroundedExample):
    """A :class:`GroundedExample` a human judged: ``grounded`` is True when every
    claim in ``answer`` is supported by ``context``."""

    grounded: bool


class GroundednessReport(BaseModel):
    """Claim-level faithfulness over a set of examples. Counts only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_examples: int = Field(..., ge=0)
    n_claims: int = Field(..., ge=0)
    unsupported_claims: int = Field(..., ge=0)
    fully_grounded_examples: int = Field(..., ge=0)
    started_at: datetime
    ended_at: datetime

    @property
    def supported_claims(self) -> int:
        return self.n_claims - self.unsupported_claims

    @property
    def faithfulness(self) -> float:
        """Supported claims / all claims. 1.0 when there were no claims at all
        (nothing to get wrong); consult ``n_claims`` to tell that from perfect."""
        return self.supported_claims / self.n_claims if self.n_claims else 1.0

    @property
    def grounded_rate(self) -> float:
        """Fraction of examples whose every claim was supported."""
        if not self.n_examples:
            return 1.0
        return self.fully_grounded_examples / self.n_examples

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "n_examples": self.n_examples,
            "n_claims": self.n_claims,
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "faithfulness": self.faithfulness,
            "fully_grounded_examples": self.fully_grounded_examples,
            "grounded_rate": self.grounded_rate,
        }


def score_groundedness(
    examples: Iterable[GroundedExample],
    guardrail: GroundednessGuardrail | None = None,
) -> GroundednessReport:
    """Score how faithful a set of generated answers is to their contexts.

    ``examples`` is consumed once. ``guardrail`` defaults to the word-overlap
    heuristic (see the module docstring for what that can and cannot see).

    Example:
        >>> ok = GroundedExample(
        ...     example_id="a",
        ...     answer="The patient has type 2 diabetes.",
        ...     context=["Assessment: type 2 diabetes mellitus, stable."],
        ... )
        >>> bad = GroundedExample(
        ...     example_id="b",
        ...     answer="The patient has a fractured femur.",
        ...     context=["Assessment: type 2 diabetes mellitus, stable."],
        ... )
        >>> report = score_groundedness([ok, bad])
        >>> report.faithfulness, report.fully_grounded_examples
        (0.5, 1)
    """
    checker = guardrail or GroundednessGuardrail()
    started_at = datetime.now(UTC)
    n = claims = unsupported = grounded = 0
    for ex in examples:
        result = checker.check(
            GroundednessCheckInput(answer=ex.answer, context=ex.context)
        )
        n += 1
        count = result.details.get("claim_count")
        if not isinstance(count, int) or isinstance(count, bool):
            raise ProcessingError(
                "The groundedness guardrail did not report details['claim_count'], "
                "so faithfulness cannot be computed. A custom guardrail must "
                "include it (the built-in GroundednessGuardrail does).",
                context={"example_id": ex.example_id},
            )
        claims += count
        bad = len(result.spans)
        unsupported += bad
        grounded += bad == 0
    return GroundednessReport(
        n_examples=n,
        n_claims=claims,
        unsupported_claims=unsupported,
        fully_grounded_examples=grounded,
        started_at=started_at,
        ended_at=datetime.now(UTC),
    )


class DetectorReport(BaseModel):
    """Agreement between the checker and human labels, for the task of
    *detecting ungrounded answers* (the positive class is "not grounded")."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    true_positives: int = Field(..., ge=0, description="Ungrounded, flagged.")
    false_positives: int = Field(..., ge=0, description="Grounded, flagged.")
    false_negatives: int = Field(..., ge=0, description="Ungrounded, missed.")
    true_negatives: int = Field(..., ge=0, description="Grounded, passed.")

    @property
    def precision(self) -> float:
        d = self.true_positives + self.false_positives
        return self.true_positives / d if d else 1.0

    @property
    def recall(self) -> float:
        d = self.true_positives + self.false_negatives
        return self.true_positives / d if d else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def evaluate_detector(
    examples: Iterable[LabelledExample],
    guardrail: GroundednessGuardrail | None = None,
) -> DetectorReport:
    """Measure how well ``guardrail`` separates grounded from ungrounded answers
    against human labels.

    Example:
        >>> good = LabelledExample(
        ...     example_id="a", grounded=True,
        ...     answer="The patient has type 2 diabetes.",
        ...     context=["Assessment: type 2 diabetes mellitus, stable."],
        ... )
        >>> bad = LabelledExample(
        ...     example_id="b", grounded=False,
        ...     answer="The patient has a fractured femur.",
        ...     context=["Assessment: type 2 diabetes mellitus, stable."],
        ... )
        >>> evaluate_detector([good, bad]).f1
        1.0
    """
    checker = guardrail or GroundednessGuardrail()
    tp = fp = fn = tn = 0
    for ex in examples:
        flagged = not checker.check(
            GroundednessCheckInput(answer=ex.answer, context=ex.context)
        ).passed
        if flagged and not ex.grounded:
            tp += 1
        elif flagged:
            fp += 1
        elif not ex.grounded:
            fn += 1
        else:
            tn += 1
    return DetectorReport(
        true_positives=tp, false_positives=fp, false_negatives=fn, true_negatives=tn
    )


def groundedness_manifest(report: GroundednessReport) -> EvalManifest:
    """The provenance record for a groundedness run (FR-X-06): when, and the
    counts -- no answer or context text."""
    return build_manifest(
        "groundedness",
        report.as_dict(),
        started_at=report.started_at,
        ended_at=report.ended_at,
    )
