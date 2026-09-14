"""Pipeline configuration: the declarative, data form of a pipeline.

A ``PipelineConfig`` is what makes a pipeline reviewable, diffable and
replayable (docs/01_VISION.md section 7.4): it is data, not code. This
module owns the schema and its loading/validation; the executor that
actually runs a config-described pipeline lives in ``openbtk.pipelines``
(a higher layer -- ``core`` does not execute anything).

Scope note: docs/04_API_DESIGN.md section 6 shows ``PipelineConfig`` with a
``provenance: ProvenanceConfig`` field and a ``guardrails: list[GuardrailConfig]``
field, but neither ``ProvenanceConfig`` nor ``GuardrailConfig`` is defined
anywhere in the docs beyond that one snippet. The only concrete evidence for
their shape is docs/03_ARCHITECTURE.md section 7.3's worked YAML example,
which is what ``ProvenanceConfig`` and ``GuardrailConfig`` below are built
from. Two things are worth flagging explicitly rather than silently
resolving:

  * That YAML example shows a guardrail's ``at`` as a **list** of attachment
    points (``[after:deid, after:chunk]``), while docs/04_API_DESIGN.md's
    Python ``.guard()`` example shows a single string (``at="after:deid"``).
    Treated the YAML as authoritative here -- checking a guardrail at
    multiple pipeline points is the more general, sensible case, and a
    single string is trivially a list of one.
  * ``PipelineConfig.validate_registry()``'s described checks
    (docs/03_ARCHITECTURE.md section 7.3) include "every params validates
    against that component's config model" and "stage types are
    compatible." Neither is implemented here: OpenBTK components take plain
    keyword arguments, not a single Pydantic config object per component
    (see core/registry.py's ``ComponentInfo`` docstring for the same
    tension), so there is no per-component schema to validate params
    against yet, and no stage-compatibility rule has been specified
    anywhere. What IS implemented: every step's ``type`` resolves in its
    registry, every ``after`` reference points at a real step, the ``after``
    graph is acyclic, and every guardrail's ``type`` resolves in
    ``GUARDRAIL_REGISTRY``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from openbtk.core.errors import ConfigError
from openbtk.core.logging import get_logger
from openbtk.core.registry import GUARDRAIL_REGISTRY, get_registry

# NOT behind TYPE_CHECKING: StepConfig.params uses this as a real Pydantic
# field type, resolved at class-definition time. Confirmed by repeating the
# exact mistake already made and fixed once in core/provenance.py -- hiding
# this import raises PydanticUserError ("not fully defined") at the first
# StepConfig construction, not at import time, so it is easy to miss until
# something is actually instantiated.
from openbtk.core.schemas import JsonValue  # noqa: TC001

log = get_logger(__name__)

_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _interpolate_env_vars(value: Any) -> Any:
    """Recursively resolve ``${VAR_NAME}`` references from the environment.

    Only string values are inspected; other types pass through unchanged.
    Applied once, at load time (``PipelineConfig.from_yaml``) -- per
    docs/06_SECURITY_COMPLIANCE.md section 3.7, the resolved value is never
    written back into a serialised config. That guarantee is the caller's
    responsibility from this point on: a loaded ``PipelineConfig`` MAY hold
    real secret values in its ``params`` fields, exactly as a caller wrote
    ``${OPENAI_API_KEY}`` intending. Never log, print, or naively
    ``.model_dump()`` a loaded config without redacting keys that look like
    credentials first (docs/06_SECURITY_COMPLIANCE.md section 3.7's
    ``key|token|secret|password|credential`` pattern) -- this module cannot
    enforce that at the type level, since by design it must actually hold
    the resolved value somewhere for the pipeline to use it.

    Raises:
        ConfigError: If a referenced environment variable is not set. Never
            silently falls back to an empty string or leaves the literal
            ``${VAR}`` placeholder in place -- both would be confusing
            failures far from their actual cause.
    """
    if isinstance(value, str):

        def _replace(match: re.Match[str]) -> str:
            var_name = match.group(1)
            if var_name not in os.environ:
                raise ConfigError(
                    f"Config references ${{{var_name}}}, but that environment "
                    "variable is not set.",
                    context={"env_var": var_name},
                )
            return os.environ[var_name]

        return _ENV_VAR_RE.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env_vars(v) for v in value]
    return value


class PolicyConfig(BaseModel):
    """Pipeline-wide safety policy.

    Example:
        >>> PolicyConfig().allow_offsite_providers
        False
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    allow_offsite_providers: bool = Field(
        False,
        description=(
            "Whether a provider with sends_data_offsite=True may be "
            "constructed. Secure default: sending PHI to a third party "
            "should require typing something (FR-V-07)."
        ),
    )
    fail_on_guardrail_block: bool = Field(
        True,
        description=(
            "Whether a BLOCK-severity guardrail result raises "
            "GuardrailViolation and halts the pipeline."
        ),
    )


class ProvenanceConfig(BaseModel):
    """Where and how this pipeline's run manifests are written.

    Minimal by design: the only field with any specification anywhere is
    ``manifest_dir``, from docs/03_ARCHITECTURE.md section 7.3's worked
    example. Extend this when core/provenance.py's RunManifest (deferred --
    see that module's docstring) actually lands and more is known about
    what a pipeline run needs to configure.

    Example:
        >>> ProvenanceConfig().manifest_dir
        './runs'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest_dir: str = Field(
        "./runs", description="Directory run manifests are written to."
    )


class GuardrailConfig(BaseModel):
    """A guardrail attached to specific points in a pipeline.

    Example:
        >>> gc = GuardrailConfig(
        ...     type="guardrail.general.phi_leakage",
        ...     at=["after:deid", "after:chunk"],
        ... )
        >>> gc.on_violation
        'block'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: str = Field(..., description="Registry key of the guardrail.")
    at: list[str] = Field(
        ...,
        min_length=1,
        description='Pipeline attachment points, e.g. "after:deid".',
    )
    on_violation: Literal["block", "warn"] = Field(
        "block",
        description=(
            "What a BLOCK-severity result from this guardrail does, "
            "overriding PolicyConfig.fail_on_guardrail_block for this "
            "guardrail specifically. Defaults to the safer option."
        ),
    )


class StepConfig(BaseModel):
    """One step in a pipeline: a registered component plus its parameters.

    Example:
        >>> StepConfig(id="load", type="loader.clinical_text.plain_text").after
        []
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(..., min_length=1, description="Unique step identifier.")
    type: str = Field(..., description="Registry key of the component.")
    params: dict[str, JsonValue] = Field(
        default_factory=dict, description="Constructor keyword arguments."
    )
    after: list[str] = Field(
        default_factory=list,
        description="Step ids this step depends on, for DAG ordering.",
    )


class ValidationIssue(BaseModel):
    """One problem found by ``PipelineConfig.validate_registry()``.

    Not specified anywhere beyond a bare return-type reference
    (docs/04_API_DESIGN.md section 6) -- designed from scratch, matching the
    shape of every other result schema in this codebase (severity, message,
    plus enough context to locate the problem).

    Example:
        >>> issue = ValidationIssue(
        ...     severity="error", message="bad step", step_id="load"
        ... )
        >>> issue.step_id
        'load'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    severity: Literal["error", "warning"] = Field(
        ..., description="Whether this issue blocks the pipeline from running."
    )
    message: str = Field(..., min_length=1, description="Human-readable description.")
    step_id: str | None = Field(
        None, description="The step this issue concerns, if applicable."
    )


class PipelineConfig(BaseModel):
    """A pipeline, described declaratively.

    Example:
        >>> cfg = PipelineConfig(
        ...     name="probe",
        ...     steps=[StepConfig(id="load", type="loader.general.probe")],
        ... )
        >>> cfg.version
        1
        >>> issues = cfg.validate_registry()
        >>> issues[0].message
        "Step 'load': type 'loader.general.probe' not found in registry 'loader'."
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1, description="Pipeline name.")
    version: int = Field(1, ge=1, description="Config schema version.")
    description: str | None = Field(None, description="Human-readable purpose.")
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    provenance: ProvenanceConfig = Field(default_factory=ProvenanceConfig)
    steps: list[StepConfig] = Field(..., min_length=1)
    guardrails: list[GuardrailConfig] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load and validate a pipeline config from a YAML file.

        Environment variable references (``${VAR_NAME}``) anywhere in the
        YAML are resolved against ``os.environ`` before validation -- see
        :func:`_interpolate_env_vars` for what that means for secret
        handling downstream.

        Raises:
            ConfigError: If the file cannot be read, is not valid YAML, an
                interpolated environment variable is unset, or the content
                does not satisfy this schema.
        """
        path = Path(path)
        try:
            raw_text = path.read_text(encoding="utf-8")
        except OSError as e:
            raise ConfigError(
                f"Could not read pipeline config: {path}",
                context={"path": str(path)},
            ) from e

        try:
            raw = yaml.safe_load(raw_text)
        except yaml.YAMLError as e:
            raise ConfigError(
                f"Invalid YAML in pipeline config: {path}",
                context={"path": str(path)},
            ) from e

        interpolated = _interpolate_env_vars(raw)

        try:
            return cls.model_validate(interpolated)
        except Exception as e:
            raise ConfigError(
                f"Pipeline config failed validation: {path}\n{e}",
                context={"path": str(path)},
            ) from e

    def validate_registry(self) -> list[ValidationIssue]:
        """Check this config against the live component registries.

        Does not instantiate anything or touch data (docs/03_ARCHITECTURE.md
        section 7.3) -- every check here is a lookup or a graph property.
        See this module's docstring for which checks from that section are
        NOT yet implemented, and why.

        Returns:
            One ``ValidationIssue`` per problem found. An empty list means
            every check that IS implemented passed -- not that the config
            is guaranteed runnable.
        """
        issues: list[ValidationIssue] = []
        step_ids = {step.id for step in self.steps}

        for step in self.steps:
            category = step.type.split(".", 1)[0]
            try:
                registry = get_registry(category)
            except Exception:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=(
                            f"Step {step.id!r}: type {step.type!r} has an "
                            f"unknown category {category!r}."
                        ),
                        step_id=step.id,
                    )
                )
                continue
            if not registry.is_registered(step.type):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=(
                            f"Step {step.id!r}: type {step.type!r} not found "
                            f"in registry {category!r}."
                        ),
                        step_id=step.id,
                    )
                )
            for dep in step.after:
                if dep not in step_ids:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            message=(
                                f"Step {step.id!r} depends on {dep!r}, which "
                                "is not a step in this pipeline."
                            ),
                            step_id=step.id,
                        )
                    )

        issues.extend(self._find_cycles(step_ids))

        for guardrail in self.guardrails:
            if not GUARDRAIL_REGISTRY.is_registered(guardrail.type):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=(
                            f"Guardrail type {guardrail.type!r} not found in "
                            "the guardrail registry."
                        ),
                    )
                )

        return issues

    def _find_cycles(self, step_ids: set[str]) -> list[ValidationIssue]:
        """Detect cycles in the ``after`` dependency graph via DFS."""
        graph = {step.id: step.after for step in self.steps}
        state: dict[str, Literal["visiting", "done"]] = {}
        issues: list[ValidationIssue] = []

        def visit(node: str) -> bool:
            if state.get(node) == "done":
                return True
            if state.get(node) == "visiting":
                return False
            state[node] = "visiting"
            for dep in graph.get(node, []):
                if dep in step_ids and not visit(dep):
                    return False
            state[node] = "done"
            return True

        for step_id in step_ids:
            if not visit(step_id):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message="The 'after' dependency graph contains a cycle.",
                        step_id=step_id,
                    )
                )
                break  # one report is enough; don't spam per node in the cycle

        return issues

    @classmethod
    def json_schema_export(cls) -> dict[str, Any]:
        """Return this config's JSON Schema, for editor validation / docs."""
        return cls.model_json_schema()
