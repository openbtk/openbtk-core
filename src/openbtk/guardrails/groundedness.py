"""``GroundednessGuardrail``: decomposes generated output into claims and
checks each against retrieved source text, returning unsupported spans
(FR-G-04) -- what makes "verifiable citations" available as a library
component (docs/03_ARCHITECTURE.md section 8.3).

**Real, disclosed scope.** Both claim decomposition and support-checking
default to small, dependency-free heuristics -- a sentence splitter and a
significant-word-overlap check -- not an NLI/entailment model or an LLM
call. This is a genuine trade-off, not a hidden one: a real entailment
check would need either a bundled model (violating the light-core rule)
or a live LLM call (an offsite dependency this guardrail should not force
on every caller). Both are overridable via constructor callables --
``decompose_claims``/``is_supported`` -- for a caller who wants a
stronger check (e.g. backed by a real ``BaseLLMProvider``), the same
dependency-injection pattern already used by
``ConceptOverlapReranker.extract_concepts`` (task 5.7).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from openbtk.core.base import BaseGuardrail
from openbtk.core.registry import GUARDRAIL_REGISTRY
from openbtk.core.schemas import GuardrailResult, GuardrailSeverity, TextSpan

if TYPE_CHECKING:
    from collections.abc import Callable

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "with",
        "this",
        "that",
        "it",
        "as",
        "by",
        "at",
        "be",
        "has",
        "have",
        "had",
        "not",
        "no",
        "her",
        "his",
        "their",
        "its",
    }
)


class GroundednessCheckInput(BaseModel):
    """The payload this guardrail's ``check()`` understands: a generated
    answer plus the source text it was supposed to be grounded in.

    Example:
        >>> GroundednessCheckInput(answer="The sky is blue.").context
        []
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    answer: str = Field(..., description="The generated text to check.")
    context: list[str] = Field(
        default_factory=list,
        description="Retrieved source texts the answer should be grounded in.",
    )


def _default_decompose_claims(text: str) -> list[str]:
    """A real, simple sentence splitter -- not sentence-boundary-detection
    grade (no abbreviation handling, no quote-nesting awareness), but
    enough to break a paragraph into claim-like units for this
    guardrail's own purpose."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _significant_words(text: str) -> set[str]:
    return {
        w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2
    }


def _default_is_supported(claim: str, context: list[str], threshold: float) -> bool:
    claim_words = _significant_words(claim)
    if not claim_words:
        return True  # nothing substantive to check (e.g. "Yes." or "Thank you.")
    context_words: set[str] = set()
    for c in context:
        context_words |= _significant_words(c)
    if not context_words:
        return False  # a substantive claim with zero context is unsupported
    overlap = claim_words & context_words
    return (len(overlap) / len(claim_words)) >= threshold


@GUARDRAIL_REGISTRY.register("guardrail.general.groundedness")
class GroundednessGuardrail(BaseGuardrail):
    """Checks whether every claim in a generated answer is supported by
    its retrieved context, returning unsupported spans.

    Args:
        support_threshold: Fraction of a claim's significant words that
            must appear in the context for the default ``is_supported``
            to count it as grounded. Ignored if ``is_supported`` is
            overridden.
        decompose_claims: ``text -> list[claim]``. Defaults to a sentence
            splitter.
        is_supported: ``(claim, context) -> bool``. Defaults to a
            significant-word-overlap heuristic against ``support_threshold``.

    ``check()`` only understands a :class:`GroundednessCheckInput`
    payload (it needs both the answer AND its context together, which no
    other schema in this project bundles) -- any other payload shape
    passes with an INFO result, the same "not my concern, not a failure"
    treatment ``TerminologyValidityGuardrail`` gives an unrelated payload.

    Example:
        >>> guardrail = GroundednessGuardrail()
        >>> grounded = GroundednessCheckInput(
        ...     answer="The patient has type 2 diabetes.",
        ...     context=["Assessment: type 2 diabetes mellitus, stable."],
        ... )
        >>> guardrail.check(grounded).passed
        True
        >>> ungrounded = GroundednessCheckInput(
        ...     answer="The patient has a fractured femur.",
        ...     context=["Assessment: type 2 diabetes mellitus, stable."],
        ... )
        >>> result = guardrail.check(ungrounded)
        >>> result.passed
        False
        >>> len(result.spans)
        1
    """

    def __init__(
        self,
        *,
        support_threshold: float = 0.5,
        decompose_claims: Callable[[str], list[str]] = _default_decompose_claims,
        is_supported: Callable[[str, list[str]], bool] | None = None,
    ) -> None:
        self._decompose_claims = decompose_claims
        self._is_supported = is_supported or (
            lambda claim, context: _default_is_supported(
                claim, context, support_threshold
            )
        )

    def check(self, payload: Any) -> GuardrailResult:
        if not isinstance(payload, GroundednessCheckInput):
            return GuardrailResult(
                passed=True,
                severity=GuardrailSeverity.INFO,
                guardrail_key=self.registry_key,
                message="Not a groundedness-checkable payload.",
            )
        claims = self._decompose_claims(payload.answer)
        if not claims:
            return GuardrailResult(
                passed=True,
                severity=GuardrailSeverity.INFO,
                guardrail_key=self.registry_key,
                message="No claims to check.",
                details={"claim_count": 0},
            )
        unsupported_spans: list[TextSpan] = []
        cursor = 0
        for claim in claims:
            start = payload.answer.find(claim, cursor)
            if start == -1:
                start, end = 0, len(payload.answer)
            else:
                end = start + len(claim)
                cursor = end
            if not self._is_supported(claim, payload.context):
                unsupported_spans.append(
                    TextSpan(
                        start=start, end=end, label="unsupported_claim", confidence=1.0
                    )
                )
        if unsupported_spans:
            return GuardrailResult(
                passed=False,
                severity=GuardrailSeverity.BLOCK,
                guardrail_key=self.registry_key,
                message=(
                    f"{len(unsupported_spans)} of {len(claims)} claim(s) are not "
                    "supported by the retrieved context."
                ),
                spans=unsupported_spans,
                details={"claim_count": len(claims)},
            )
        return GuardrailResult(
            passed=True,
            severity=GuardrailSeverity.INFO,
            guardrail_key=self.registry_key,
            message=f"All {len(claims)} claim(s) are supported.",
            details={"claim_count": len(claims)},
        )
