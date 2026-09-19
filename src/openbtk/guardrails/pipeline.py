"""``GuardrailPipeline``: runs guardrails in sequence over one payload,
aggregates results, and short-circuits on the first ``BLOCK`` when
configured (FR-G-06), recording every outcome in a shape a caller can
attach to a run manifest (FR-G-07).

**Not registered in ``GUARDRAIL_REGISTRY``.** It composes OTHER
guardrails -- by registry key or real instance -- the same "this
component's entire configuration IS other components" treatment already
given to ``CachedTerminologyService`` (task 7.1): nothing here is itself
a single check, so it does not belong in a registry of checks.

**A separate thing from the streaming executor's own per-step guardrail
attachment** (``Pipeline.guard(key, at=...)``, ``openbtk.pipelines.executor``):
that mechanism runs one guardrail at a time, once per record, over an
entire streamed run, and aggregates into ``GuardrailOutcome`` itself.
``GuardrailPipeline`` is for the other real, common case: checking
several guardrails together against a single payload OUTSIDE any
streaming pipeline run at all -- e.g. validating one LLM response before
it is returned to a caller. ``to_guardrail_outcomes()`` lets a caller who
DOES want manifest-shaped provenance for that check get it, reusing the
exact same ``GuardrailOutcome`` schema the executor emits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from openbtk.core.provenance import GuardrailOutcome
from openbtk.core.registry import GUARDRAIL_REGISTRY
from openbtk.core.schemas import GuardrailResult, GuardrailSeverity

if TYPE_CHECKING:
    from collections.abc import Sequence

    from openbtk.core.base import BaseGuardrail
    from openbtk.core.config import PolicyConfig


class GuardrailPipelineResult(BaseModel):
    """The outcome of one :meth:`GuardrailPipeline.run` call.

    Example:
        >>> result = GuardrailPipelineResult(
        ...     passed=True,
        ...     results=[GuardrailResult(
        ...         passed=True, severity=GuardrailSeverity.INFO,
        ...         guardrail_key="guardrail.general.phi_leakage",
        ...         message="No PHI detected.",
        ...     )],
        ... )
        >>> result.short_circuited
        False
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool = Field(
        ..., description="False if any result was a failed BLOCK-severity check."
    )
    results: list[GuardrailResult] = Field(
        ..., description="One result per guardrail actually run, in order."
    )
    short_circuited: bool = Field(
        False, description="True if a BLOCK stopped later guardrails from running."
    )

    def to_guardrail_outcomes(self, at: str) -> list[GuardrailOutcome]:
        """Reshape each result into a ``GuardrailOutcome`` (``checked_count=1``
        each, since this pipeline checks one payload, not a whole
        streamed run) for attaching to a ``RunManifest``.

        Args:
            at: An attachment-point-shaped label the caller chooses --
                this pipeline has no step id of its own, unlike the
                executor's own guardrail attachment.
        """
        return [
            GuardrailOutcome(
                guardrail_key=r.guardrail_key,
                at=at,
                checked_count=1,
                blocked_count=int(
                    not r.passed and r.severity is GuardrailSeverity.BLOCK
                ),
                warned_count=int(
                    not r.passed and r.severity is GuardrailSeverity.WARNING
                ),
                sample_messages=[] if r.passed else [r.message],
            )
            for r in self.results
        ]


class GuardrailPipeline:
    """Run an ordered list of guardrails over one payload.

    Args:
        guardrails: Each entry is either a real ``BaseGuardrail`` instance
            or a registry key string (constructed via
            ``GUARDRAIL_REGISTRY.create``).
        policy: Forwarded to ``GUARDRAIL_REGISTRY.create`` for any
            string entries -- irrelevant for guardrails that don't send
            data offsite (true of every guardrail this project ships).
        short_circuit: When True (the default), stop running further
            guardrails once one returns a failed BLOCK-severity result.
            When False, every guardrail always runs, for a complete
            diagnostic report rather than a fail-fast check.

    Example:
        >>> import contextlib, io
        >>> with contextlib.redirect_stdout(io.StringIO()):
        ...     from openbtk.guardrails.phi_leakage import PHILeakageGuardrail
        ...     pipeline = GuardrailPipeline([PHILeakageGuardrail()])
        ...     result = pipeline.run("Patient SSN: 123-45-6789.")
        >>> result.passed
        False
        >>> len(result.results)
        1
    """

    def __init__(
        self,
        guardrails: Sequence[BaseGuardrail | str],
        *,
        policy: PolicyConfig | None = None,
        short_circuit: bool = True,
    ) -> None:
        self._guardrails = [_resolve(g, policy) for g in guardrails]
        self._short_circuit = short_circuit

    def run(self, payload: Any) -> GuardrailPipelineResult:
        results: list[GuardrailResult] = []
        short_circuited = False
        for guardrail in self._guardrails:
            result = guardrail.check(payload)
            results.append(result)
            if (
                self._short_circuit
                and not result.passed
                and result.severity is GuardrailSeverity.BLOCK
            ):
                short_circuited = True
                break
        overall_passed = not any(
            not r.passed and r.severity is GuardrailSeverity.BLOCK for r in results
        )
        return GuardrailPipelineResult(
            passed=overall_passed, results=results, short_circuited=short_circuited
        )


def _resolve(
    guardrail: BaseGuardrail | str, policy: PolicyConfig | None
) -> BaseGuardrail:
    if isinstance(guardrail, str):
        return GUARDRAIL_REGISTRY.create(guardrail, policy=policy)
    return guardrail
