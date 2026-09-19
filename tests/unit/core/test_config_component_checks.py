"""``PipelineConfig.validate_registry()``'s component-level checks (M10 10.1):
parameters against the constructor signature, and the off-site policy --
both without instantiating anything or touching data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openbtk import embeddings as _embeddings  # noqa: F401 -- registers embedding.*
from openbtk import llms as _llms  # noqa: F401 -- registers llm.*
from openbtk.core.base import BaseLoader
from openbtk.core.config import (
    PipelineConfig,
    PolicyConfig,
    StepConfig,
    _component_issues,
)
from openbtk.core.registry import Registry
from openbtk.data import clinical_text as _clinical_text  # noqa: F401

if TYPE_CHECKING:
    from collections.abc import Iterator


def _config(*steps: StepConfig, policy: PolicyConfig | None = None) -> PipelineConfig:
    return PipelineConfig(name="t", steps=list(steps), policy=policy or PolicyConfig())


def _errors(config: PipelineConfig) -> list[str]:
    return [i.message for i in config.validate_registry() if i.severity == "error"]


class TestUnknownParameters:
    def test_a_parameter_the_constructor_does_not_take_is_an_error(self) -> None:
        config = _config(
            StepConfig(id="load", type="loader.clinical_text.plain_text"),
            StepConfig(
                id="chunk",
                type="chunker.clinical_text.section_aware",
                params={"max_tokens": 100, "max_token": 5},
                after=["load"],
            ),
        )
        (message,) = _errors(config)
        assert "'max_token'" in message and "'chunk'" in message
        assert "max_tokens" in message  # names what IS accepted

    def test_known_parameters_pass(self) -> None:
        config = _config(
            StepConfig(id="load", type="loader.clinical_text.plain_text"),
            StepConfig(
                id="chunk",
                type="chunker.clinical_text.section_aware",
                params={"max_tokens": 100},
                after=["load"],
            ),
        )
        assert _errors(config) == []

    def test_a_loaders_path_and_source_are_load_arguments_not_constructor_ones(
        self,
    ) -> None:
        for key in ("path", "source"):
            config = _config(
                StepConfig(
                    id="load",
                    type="loader.clinical_text.plain_text",
                    params={key: "./notes"},
                )
            )
            assert _errors(config) == []

    def test_a_loader_still_rejects_other_unknown_parameters(self) -> None:
        config = _config(
            StepConfig(
                id="load",
                type="loader.clinical_text.plain_text",
                params={"path": "./n", "encodng": "utf-8"},
            )
        )
        assert any("'encodng'" in m for m in _errors(config))


class _NeedsPath(BaseLoader[str, Any]):
    def __init__(self, *, path: str, mode: str = "r") -> None:
        self._path = path
        self._mode = mode

    def load(self, source: str) -> Iterator[Any]:
        return iter(())


class _TakesAnything(BaseLoader[str, Any]):
    def __init__(self, **options: Any) -> None:
        self._options = options

    def load(self, source: str) -> Iterator[Any]:
        return iter(())


def _private_registry() -> Registry[Any]:
    registry: Registry[Any] = Registry("loader", BaseLoader)  # type: ignore[type-abstract]
    registry.register("loader.general.needs_path")(_NeedsPath)
    registry.register("loader.general.takes_anything")(_TakesAnything)
    return registry


class TestRequiredAndOpenSignatures:
    def test_a_missing_required_parameter_is_an_error(self) -> None:
        step = StepConfig(id="s", type="loader.general.needs_path")
        issues = _component_issues(step, _private_registry(), PolicyConfig())
        assert [i.message for i in issues] == [
            "Step 's': 'loader.general.needs_path' requires the parameter 'path'."
        ]

    def test_supplying_it_clears_the_error(self) -> None:
        step = StepConfig(
            id="s", type="loader.general.needs_path", params={"path": "x", "mode": "rb"}
        )
        assert _component_issues(step, _private_registry(), PolicyConfig()) == []

    def test_a_constructor_taking_kwargs_accepts_any_parameter(self) -> None:
        step = StepConfig(
            id="s", type="loader.general.takes_anything", params={"whatever": 1}
        )
        assert _component_issues(step, _private_registry(), PolicyConfig()) == []


class TestOffsitePolicy:
    def _llm_config(self, policy: PolicyConfig) -> PipelineConfig:
        return _config(StepConfig(id="llm", type="llm.general.openai"), policy=policy)

    def test_an_offsite_provider_is_refused_at_validation_under_the_default_policy(
        self,
    ) -> None:
        (message,) = _errors(self._llm_config(PolicyConfig()))
        assert "sends data offsite" in message
        assert "OpenAIProvider" in message
        assert "llm.general.openai" in message

    def test_an_explicit_opt_in_clears_it(self) -> None:
        config = self._llm_config(PolicyConfig(allow_offsite_providers=True))
        assert not any("offsite" in m for m in _errors(config))

    def test_a_local_component_is_never_flagged(self) -> None:
        config = _config(StepConfig(id="load", type="loader.clinical_text.plain_text"))
        assert not any("offsite" in m for m in _errors(config))

    def test_the_embedding_side_is_checked_too(self) -> None:
        config = _config(StepConfig(id="e", type="embedding.general.openai"))
        assert any("sends data offsite" in m for m in _errors(config))
