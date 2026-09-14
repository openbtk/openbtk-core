"""Provenance primitives -- identity and traceability, not an afterthought.

This module landed in two increments:

  * **M1:** ``ModelIdentity`` and ``ComponentProvenance``. These are what
    ``core.base.Component`` needs, so they had to exist before ``base.py``
    does -- every base class exposes ``provenance()``.
  * **M3 (task 3.7):** ``RunManifest`` and its remaining parts
    (``StepProvenance``, ``DataDigest``, ``GuardrailOutcome``,
    ``TokenUsage``), now that ``openbtk.pipelines.executor`` (the thing
    that actually produces one) exists to validate the shape against.

**``RunManifest.config`` is a serialised snapshot (``dict[str, JsonValue]``
via ``PipelineConfig.model_dump(mode="json")``), not a ``PipelineConfig``
object.** Two reasons, not one: (1) a manifest wants a frozen fact about
what ran, not a live, re-validatable object a caller could be tempted to
mutate and re-run; (2) typing it as ``PipelineConfig`` would make
``core.provenance`` import ``core.config``, which imports ``core.registry``,
which imports ``core.base``, which imports ``core.provenance`` -- a real
cycle, not a hypothetical one, caught by tracing the actual import graph
rather than assumed away.

**Guardrail outcomes are aggregated per (guardrail, attachment point), not
one entry per record.** docs/03_ARCHITECTURE.md section 8.3 says a
``GuardrailPipeline`` "writes every outcome to the run manifest" -- read
literally, that is one entry per record, which for a corpus of millions
would make the manifest itself violate NFR-01's O(batch) memory guarantee
(and blow past "manifests contain no PHI" into "manifests contain
everything"). ``GuardrailOutcome`` instead carries checked/blocked/warned
counts plus a small, bounded sample of messages -- real, aggregate
provenance, not a per-record log. A disclosed, deliberate reading of an
under-specified line, not a silent shortcut.

**``TokenUsage`` is declared but never populated yet.** No LLM-provider
step exists in the executor's dispatch table as of task 3.7 (no real
``llms``/``embeddings`` component exists in this repository yet either) --
the schema exists now because ADR-0005 names it as part of this module's
required primitive set, ready for the executor to populate once an LLM
step is real.

See ADR-0005 (provenance as a core primitive) for why this exists at all.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 -- RunManifest field type, resolved eagerly
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# NOT behind TYPE_CHECKING (despite what ruff's TC001 suggests): Pydantic
# resolves this field annotation at class-definition time. Verified directly --
# hiding this import behind TYPE_CHECKING raises PydanticUserError because
# JsonValue would not exist in the module's runtime namespace when the model
# schema is built.
from openbtk.core.schemas import JsonValue  # noqa: TC001

RunStatus = Literal["success", "failed", "partial"]
"""ADR-0005: "success, failure or partial." The executor (task 3.7) only
ever produces "success" or "failed" today -- "partial" is reserved for a
future resumable/checkpointed run, not yet implemented, and is declared
here rather than omitted so the schema does not need a breaking change
when that lands."""

# Revision strings that name a moving target rather than an immutable point in
# history. FR-P-05 (docs/02_PRD.md): model provenance must record an HF
# revision SHA or a pinned API model version, "never a floating tag" -- this
# is the enforced form of that requirement, not just a comment on the field.
_FLOATING_TAGS = frozenset({"latest", "main", "master", "head", "stable"})


class ModelIdentity(BaseModel):
    """The identity of a specific model, pinned to an immutable revision.

    Example:
        >>> ModelIdentity(
        ...     name="NeuML/pubmedbert-base-embeddings",
        ...     revision="a1b2c3d4e5f6",
        ...     source="huggingface",
        ... ).source
        'huggingface'
        >>> ModelIdentity(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     name="gpt-4o", revision="latest", source="api"
        ... )
        Traceback (most recent call last):
            ...
        pydantic_core._pydantic_core.ValidationError: ...revision is a floating tag
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(
        ..., min_length=1, description="Model name or identifier, e.g. a HF repo id."
    )
    revision: str = Field(
        ...,
        min_length=1,
        description=(
            "An immutable point in history: a commit SHA for HuggingFace models, "
            "or a pinned API model version. Never a floating tag such as 'latest'."
        ),
    )
    source: str = Field(
        ...,
        min_length=1,
        description='Where this model comes from, e.g. "huggingface", "api", "local".',
    )

    @field_validator("revision")
    @classmethod
    def _reject_floating_tag(cls, value: str) -> str:
        """Enforce FR-P-05: revisions must be immutable, not a moving target."""
        if value.strip().lower() in _FLOATING_TAGS:
            raise ValueError(
                f"revision={value!r} is a floating tag, not an immutable revision. "
                "Use a commit SHA or a pinned API model version instead."
            )
        return value


class ComponentProvenance(BaseModel):
    """Identity and configuration of one component instance, for a run manifest.

    Every ``Component`` exposes a default ``provenance()`` that builds one of
    these from class name, registry key and package version alone. A
    component whose behaviour depends on a specific model overrides
    ``provenance()`` to populate ``model_identity`` too.

    Example:
        >>> cp = ComponentProvenance(
        ...     registry_key="loader.clinical_text.plain_text",
        ...     class_name="PlainTextLoader",
        ...     package_version="0.1.0",
        ...     config={"encoding": "utf-8"},
        ... )
        >>> cp.model_identity is None
        True
        >>> ComponentProvenance(
        ...     registry_key="", class_name="Unregistered", package_version="0.1.0"
        ... ).registry_key
        ''
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    registry_key: str = Field(
        ...,
        description=(
            "The registry key this component was created under, or an empty "
            "string if the component was constructed without going through "
            "the registry (e.g. a test fixture)."
        ),
    )
    class_name: str = Field(..., min_length=1, description="The concrete class name.")
    package_version: str = Field(
        ..., min_length=1, description="openbtk version that produced this component."
    )
    config: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Constructor configuration, with secrets redacted.",
    )
    model_identity: ModelIdentity | None = Field(
        None, description="The specific model this component wraps, if any."
    )


class DataDigest(BaseModel):
    """A content digest for one input source, for a run manifest.

    Example:
        >>> DataDigest(uri="./notes", sha256=None, record_count=3).record_count
        3
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    uri: str = Field(
        ...,
        min_length=1,
        description="Source identifier -- a path or URI, never content.",
    )
    sha256: str | None = Field(
        None,
        description=(
            "SHA-256 of the source content, when it was practical to compute "
            "(a single, existing, bounded-size local file). None -- not a "
            "fabricated value -- when the source is a directory, does not "
            "exist as a plain file, or exceeds the size cap the executor "
            "applies to keep hashing from breaking the streaming guarantee "
            "for a large corpus."
        ),
    )
    record_count: int = Field(..., ge=0, description="Records read from this source.")


class TokenUsage(BaseModel):
    """Token accounting for a run. See this module's docstring: declared,
    not yet populated by anything in this repository.

    Example:
        >>> TokenUsage().total_tokens
        0
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_tokens: int = Field(0, ge=0)
    completion_tokens: int = Field(0, ge=0)
    total_tokens: int = Field(0, ge=0)


class GuardrailOutcome(BaseModel):
    """Aggregated outcome of one guardrail at one attachment point over an
    entire run. See this module's docstring for why this is aggregated
    rather than one entry per record.

    Example:
        >>> outcome = GuardrailOutcome(
        ...     guardrail_key="guardrail.general.phi_leakage",
        ...     at="after:deid",
        ...     checked_count=100,
        ...     blocked_count=0,
        ... )
        >>> outcome.warned_count
        0
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    guardrail_key: str = Field(..., min_length=1)
    at: str = Field(
        ..., min_length=1, description='Attachment point, e.g. "after:deid".'
    )
    checked_count: int = Field(..., ge=0)
    blocked_count: int = Field(0, ge=0)
    warned_count: int = Field(0, ge=0)
    sample_messages: list[str] = Field(
        default_factory=list,
        description=(
            "A small, bounded sample of PHI-free GuardrailResult messages "
            "from failed checks, for diagnosis -- never one per record."
        ),
    )


class StepProvenance(BaseModel):
    """Identity, throughput and outcome of one executed pipeline step.

    Example:
        >>> sp = StepProvenance(
        ...     step_id="load",
        ...     component=ComponentProvenance(
        ...         registry_key="loader.clinical_text.plain_text",
        ...         class_name="PlainTextLoader",
        ...         package_version="0.1.0",
        ...     ),
        ...     records_in=0,
        ...     records_out=3,
        ...     status="success",
        ... )
        >>> sp.records_out
        3
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_id: str = Field(..., min_length=1)
    component: ComponentProvenance
    records_in: int = Field(
        ...,
        ge=0,
        description=(
            "Items consumed from the predecessor stream (0 for a root loader step)."
        ),
    )
    records_out: int = Field(
        ...,
        ge=0,
        description=(
            "Items yielded downstream. Differs from records_in for a "
            "fan-out step (a chunker producing many chunks per record)."
        ),
    )
    status: Literal["success", "failed"] = Field(...)
    error: str | None = Field(
        None, description="PHI-free failure summary, if status == 'failed'."
    )


class RunManifest(BaseModel):
    """The audit record of one pipeline execution (ADR-0005). Emitted by
    every run -- success, failure or partial -- there is no manifest-off
    switch.

    Example:
        >>> from datetime import UTC, datetime
        >>> manifest = RunManifest(
        ...     run_id="a1b2c3",
        ...     status="success",
        ...     config={"name": "probe", "steps": []},
        ...     started_at=datetime(2026, 1, 1, tzinfo=UTC),
        ... )
        >>> manifest.manifest_version
        '1.0.0'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest_version: str = Field(
        "1.0.0",
        description=(
            "Schema version, independent of the openbtk package version "
            "(ADR-0005): additive-only within a major."
        ),
    )
    run_id: str = Field(..., min_length=1)
    status: RunStatus = Field(...)
    config: dict[str, JsonValue] = Field(
        ...,
        description=(
            "A serialised snapshot of the PipelineConfig that produced "
            "this run (PipelineConfig.model_dump(mode='json')) -- see this "
            "module's docstring for why this is a plain dict, not a "
            "PipelineConfig, here."
        ),
    )
    started_at: datetime
    ended_at: datetime | None = None
    steps: list[StepProvenance] = Field(default_factory=list)
    input_digests: list[DataDigest] = Field(default_factory=list)
    guardrail_outcomes: list[GuardrailOutcome] = Field(default_factory=list)
    token_usage: TokenUsage | None = None
    error: str | None = Field(
        None, description="PHI-free top-level failure summary, if status == 'failed'."
    )
