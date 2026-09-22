"""Model cards for the models a pipeline wraps or you fine-tuned (FR-P-06).

A *model card* (Mitchell et al., 2019) is the short document that travels with a model:
what it is, what it is for, how well it does, and where it fails. OpenBTK already
records the facts a card needs, namely the exact model and revision a component used
(:class:`~openbtk.core.provenance.ModelIdentity`) and the scores an evaluation produced
(:class:`~openbtk.eval.manifest.EvalManifest`). This module assembles them.

**What is written for you, and what is not.** The *identity* section (name, pinned
revision, source, the OpenBTK component and its version) and the *evaluation* section
come from those records. Everything that needs a human judgement, namely intended use,
out-of-scope use, training data, limitations and ethical considerations, is text **you**
supply, and a section you leave out reads "Not provided." A card never states something
OpenBTK does not
know: it does not describe a model's training data, invent limitations or praise it.

**No number without its harness** (CLAUDE.md rule 14). A metric enters a card only
through :meth:`ModelCard.with_evaluation`, which reads it from an ``EvalManifest`` and
records that manifest's id, kind and input digests beside it. There is no way to type a
score into a card. A card with no evaluations says so.

The card is a frozen pydantic model, so it serialises to JSON for machine use;
:meth:`ModelCard.to_markdown` renders the human version, and :meth:`ModelCard.write`
saves it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from openbtk.core.errors import ConfigError
from openbtk.core.provenance import ComponentProvenance, ModelIdentity
from openbtk.core.schemas import JsonValue  # noqa: TC001

if TYPE_CHECKING:
    from openbtk.core.base import Component
    from openbtk.core.provenance import RunManifest
    from openbtk.eval.manifest import EvalManifest

_NOT_PROVIDED = "Not provided."


class CardEvaluation(BaseModel):
    """One metric on a card, with the evaluation run it came from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str = Field(..., min_length=1, description="What the number is called.")
    value: float
    eval_id: str = Field(..., description="The ``EvalManifest`` it was read from.")
    kind: str = Field(..., description="The evaluation's kind, e.g. ``deid``.")
    inputs: list[str] = Field(
        default_factory=list,
        description="``uri`` and SHA-256 of each input the evaluation ran on.",
    )


class ModelCard(BaseModel):
    """A model card. Build one with :meth:`for_component` or :func:`cards_from_run`.

    Example:
        >>> from openbtk.core.provenance import ModelIdentity
        >>> card = ModelCard(
        ...     model=ModelIdentity(
        ...         name="org/model", revision="abc1234", source="huggingface"
        ...     ),
        ...     intended_use="Ranking passages for clinical question answering.",
        ... )
        >>> "## Intended use" in card.to_markdown()
        True
        >>> "Not provided." in card.to_markdown()  # the sections you did not write
        True
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: ModelIdentity
    component_key: str | None = Field(
        None, description="The OpenBTK registry key that used the model."
    )
    openbtk_version: str | None = None
    settings: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="The component's configuration (secrets are never recorded there).",
    )
    description: str | None = None
    intended_use: str | None = None
    out_of_scope_use: str | None = None
    training_data: str | None = None
    limitations: str | None = None
    ethical_considerations: str | None = None
    contact: str | None = None
    evaluations: list[CardEvaluation] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # ------------------------------------------------------------ construction

    @classmethod
    def for_component(
        cls,
        component: Component | ComponentProvenance,
        *,
        description: str | None = None,
        intended_use: str | None = None,
        out_of_scope_use: str | None = None,
        training_data: str | None = None,
        limitations: str | None = None,
        ethical_considerations: str | None = None,
        contact: str | None = None,
    ) -> ModelCard:
        """A card for the model a component wraps.

        Args:
            component: A component instance or its provenance. It must carry a model
                identity (an LLM, an embedding provider, a cross-encoder reranker, ...).
            description: One or two sentences on what the model is.
            intended_use: What it is meant for.
            out_of_scope_use: What it must not be used for.
            training_data: What it was trained on (for a fine-tune, what *you* trained
                it on). OpenBTK cannot know this and does not guess.
            limitations: Known failure modes and blind spots.
            ethical_considerations: Risks, populations affected, mitigations.
            contact: Who to ask.

        Raises:
            ConfigError: If the component has no model identity (it wraps no model).
        """
        provenance = (
            component
            if isinstance(component, ComponentProvenance)
            else component.provenance()
        )
        if provenance.model_identity is None:
            raise ConfigError(
                f"{provenance.registry_key or provenance.class_name} wraps no model, "
                "so there is no model card to write.",
                context={"component": provenance.registry_key},
            )
        return cls(
            model=provenance.model_identity,
            component_key=provenance.registry_key or None,
            openbtk_version=provenance.package_version,
            settings=provenance.config,
            description=description,
            intended_use=intended_use,
            out_of_scope_use=out_of_scope_use,
            training_data=training_data,
            limitations=limitations,
            ethical_considerations=ethical_considerations,
            contact=contact,
        )

    def with_evaluation(
        self, manifest: EvalManifest, metric: str, *, name: str | None = None
    ) -> ModelCard:
        """A copy of this card with one score from an evaluation run attached.

        Args:
            manifest: The ``EvalManifest`` the evaluation produced.
            metric: The key of the score in ``manifest.report``, e.g. ``"f1"``.
            name: How to label it on the card (default: ``metric``).

        Raises:
            ConfigError: If the report has no numeric value under ``metric``. A card
                never records a number the harness did not produce.
        """
        value = manifest.report.get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(
                f"The evaluation report has no numeric {metric!r} to put on the card.",
                context={"eval_id": manifest.eval_id, "metric": metric},
            )
        entry = CardEvaluation(
            metric=name or metric,
            value=float(value),
            eval_id=manifest.eval_id,
            kind=manifest.kind,
            inputs=[
                f"{d.uri} (sha256 {d.sha256})" if d.sha256 else f"{d.uri} (not hashed)"
                for d in manifest.input_digests
            ],
        )
        return self.model_copy(update={"evaluations": [*self.evaluations, entry]})

    # --------------------------------------------------------------- rendering

    def to_markdown(self) -> str:
        """The card as Markdown. Sections you did not provide say so."""

        def section(title: str, body: str | None) -> str:
            return f"## {title}\n\n{body or _NOT_PROVIDED}\n"

        identity = [
            f"- **Model:** `{self.model.name}`",
            f"- **Revision:** `{self.model.revision}`",
            f"- **Source:** {self.model.source}",
        ]
        if self.component_key:
            identity.append(f"- **OpenBTK component:** `{self.component_key}`")
        if self.openbtk_version:
            identity.append(f"- **OpenBTK version:** {self.openbtk_version}")
        parts = [
            f"# Model card: {self.model.name}\n",
            "## Model details\n\n" + "\n".join(identity) + "\n",
            section("Description", self.description),
            section("Intended use", self.intended_use),
            section("Out-of-scope use", self.out_of_scope_use),
            section("Training data", self.training_data),
            self._evaluation_section(),
            section("Limitations", self.limitations),
            section("Ethical considerations", self.ethical_considerations),
            section("Contact", self.contact),
        ]
        if self.settings:
            rows = "\n".join(
                f"- `{k}`: `{v}`" for k, v in sorted(self.settings.items())
            )
            parts.append(f"## Configuration\n\n{rows}\n")
        parts.append(f"---\nGenerated {self.generated_at.date().isoformat()}.\n")
        return "\n".join(parts)

    def _evaluation_section(self) -> str:
        if not self.evaluations:
            return (
                "## Evaluation\n\nNo evaluation is attached to this card. A number "
                "appears here only when it comes from an evaluation run.\n"
            )
        lines = ["| Metric | Value | Evaluation | Inputs |", "|---|---|---|---|"]
        for e in self.evaluations:
            inputs = "<br>".join(e.inputs) or "none recorded"
            lines.append(
                f"| {e.metric} | {e.value:.4g} | `{e.eval_id}` ({e.kind}) | {inputs} |"
            )
        return "## Evaluation\n\n" + "\n".join(lines) + "\n"

    def write(self, path: str | Path) -> Path:
        """Write the Markdown card to ``path`` (creating parent directories)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_markdown(), encoding="utf-8")
        return target


def cards_from_run(manifest: RunManifest, **text: str | None) -> list[ModelCard]:
    """One card per distinct model a run used, read from its manifest.

    Every step whose component recorded a model identity contributes a card; a model
    used by several steps appears once. Any keyword (``intended_use=...``,
    ``limitations=...``) is applied to every card, so pass only what is true of all of
    them, and write the rest with :meth:`ModelCard.for_component`.

    """
    cards: dict[tuple[str, str], ModelCard] = {}
    for step in manifest.steps:
        identity = step.component.model_identity
        if identity is None:
            continue
        key = (identity.name, identity.revision)
        if key not in cards:
            cards[key] = ModelCard.for_component(step.component, **text)
    return list(cards.values())
