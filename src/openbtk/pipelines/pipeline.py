"""``Step`` and ``Pipeline``: the public builder API (docs/04_API_DESIGN.md
section 6). Both the programmatic (``Pipeline(...).add(Step(...))``) and
declarative (``Pipeline.from_yaml(...)``) surfaces converge on the same
``PipelineConfig``, so ``openbtk.pipelines.executor`` has exactly one shape
to execute regardless of which surface built it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from openbtk.core.config import (
    GuardrailConfig,
    PipelineConfig,
    PolicyConfig,
    ProvenanceConfig,
    StepConfig,
)
from openbtk.pipelines.executor import _Executor

if TYPE_CHECKING:
    from pathlib import Path

    from openbtk.core.provenance import RunManifest


class Step:
    """One pipeline step, for the programmatic builder API.

    Example:
        >>> step = Step("load", "loader.clinical_text.plain_text", path="./notes")
        >>> step.params
        {'path': './notes'}
    """

    def __init__(
        self, step_id: str, type: str, *, after: list[str] | None = None, **params: Any
    ) -> None:
        self.id = step_id
        self.type = type
        self.after = after
        self.params: dict[str, Any] = params


class Pipeline:
    """Build and run a streaming pipeline.

    Example:
        >>> pipeline = Pipeline("probe").add(
        ...     Step("load", "loader.general.contract_reference")
        ... )
        >>> pipeline.to_config().name
        'probe'
    """

    def __init__(
        self,
        name: str,
        *,
        version: int = 1,
        description: str | None = None,
        policy: PolicyConfig | None = None,
        provenance: ProvenanceConfig | None = None,
    ) -> None:
        self._name = name
        self._version = version
        self._description = description
        self._policy = policy if policy is not None else PolicyConfig()
        self._provenance = provenance if provenance is not None else ProvenanceConfig()
        self._steps: list[Step] = []
        self._guardrails: list[GuardrailConfig] = []

    def add(self, step: Step) -> Pipeline:
        """Append a step. A step with no explicit ``after`` chains onto the
        previously added step -- the common, linear case needs no
        boilerplate; a config loaded from YAML always states ``after``
        explicitly instead (``StepConfig.after`` defaults to ``[]``, not
        "infer from position")."""
        after = step.after
        if after is None:
            after = [self._steps[-1].id] if self._steps else []
        resolved = Step(step.id, step.type, after=after, **step.params)
        self._steps.append(resolved)
        return self

    def guard(
        self,
        guardrail_type: str,
        *,
        at: str | list[str],
        on_violation: Literal["block", "warn"] = "block",
    ) -> Pipeline:
        """Attach a guardrail at one or more ``"after:<step_id>"`` points."""
        at_list = [at] if isinstance(at, str) else list(at)
        self._guardrails.append(
            GuardrailConfig(type=guardrail_type, at=at_list, on_violation=on_violation)
        )
        return self

    def to_config(self) -> PipelineConfig:
        """The declarative ``PipelineConfig`` this builder currently describes."""
        return PipelineConfig(
            name=self._name,
            version=self._version,
            description=self._description,
            policy=self._policy,
            provenance=self._provenance,
            steps=[
                StepConfig(id=s.id, type=s.type, params=s.params, after=s.after or [])
                for s in self._steps
            ],
            guardrails=self._guardrails,
        )

    @classmethod
    def from_config(cls, config: PipelineConfig) -> Pipeline:
        pipeline = cls(
            config.name,
            version=config.version,
            description=config.description,
            policy=config.policy,
            provenance=config.provenance,
        )
        pipeline._steps = [
            Step(s.id, s.type, after=s.after, **s.params) for s in config.steps
        ]
        pipeline._guardrails = list(config.guardrails)
        return pipeline

    @classmethod
    def from_yaml(cls, path: str | Path) -> Pipeline:
        return cls.from_config(PipelineConfig.from_yaml(path))

    def run(self) -> RunManifest:
        """Execute this pipeline, streaming records through every step.

        Always returns a ``RunManifest`` -- success or failure (ADR-0005:
        there is no manifest-off switch). Never raises: a
        ``GuardrailViolation`` or any ``OpenBTKError`` raised during wiring
        or execution is caught and reported as
        ``RunManifest.status == "failed"`` with a PHI-free
        ``RunManifest.error`` instead.
        """
        return _Executor(self.to_config()).run()
