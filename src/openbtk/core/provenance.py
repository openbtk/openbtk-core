"""Provenance primitives -- identity and traceability, not an afterthought.

This module is delivered in two increments, and that split is deliberate
rather than an oversight:

  * **This increment:** ``ModelIdentity`` and ``ComponentProvenance``. These
    are what ``core.base.Component`` needs, so they must exist before
    ``base.py`` does -- every base class exposes ``provenance()``.
  * **Next increment (M1.7 proper):** ``RunManifest`` and its remaining
    parts (``StepProvenance``, ``DataDigest``, ``GuardrailOutcome``,
    ``TokenUsage``). ``RunManifest.config`` is a ``PipelineConfig``
    (docs/03_ARCHITECTURE.md section 4.4), which does not exist until
    ``core/config.py`` (task 1.8) is written. Writing a placeholder
    ``RunManifest`` now, ahead of the type it must embed, would mean
    revisiting it twice for no benefit -- the roadmap's task numbering
    (1.7 before 1.8) does not reflect this dependency, so this docstring
    records the actual order rather than silently deviating from it.

See ADR-0005 (provenance as a core primitive) for why this exists at all.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

# NOT behind TYPE_CHECKING (despite what ruff's TC001 suggests): Pydantic
# resolves this field annotation at class-definition time. Verified directly --
# hiding this import behind TYPE_CHECKING raises PydanticUserError because
# JsonValue would not exist in the module's runtime namespace when the model
# schema is built.
from openbtk.core.schemas import JsonValue  # noqa: TC001

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
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    registry_key: str = Field(
        ...,
        min_length=1,
        description="The registry key this component was created under.",
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
