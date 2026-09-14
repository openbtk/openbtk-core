"""Unit tests for openbtk.pipelines.pipeline: Step and Pipeline's builder
API (to_config/from_config/from_yaml, add()'s chaining default, guard()).

Execution itself (run()) is openbtk.pipelines.executor's own concern,
covered by tests/unit/pipelines/test_executor.py -- this file is only
about the shape of the PipelineConfig a builder produces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from openbtk.core.config import PipelineConfig, PolicyConfig, ProvenanceConfig
from openbtk.core.errors import ConfigError
from openbtk.pipelines import Pipeline, Step

if TYPE_CHECKING:
    from pathlib import Path


class TestStep:
    def test_params_are_the_keyword_arguments(self) -> None:
        step = Step(
            "load", "loader.clinical_text.plain_text", path="./notes", note_type="x"
        )
        assert step.params == {"path": "./notes", "note_type": "x"}

    def test_after_defaults_to_none_not_empty_list(self) -> None:
        """None is the "infer from Pipeline.add() position" sentinel --
        distinct from an explicit empty list, which StepConfig itself uses
        to mean "no dependencies, and don't infer any."""
        assert Step("a", "loader.general.x").after is None


class TestPipelineAddChaining:
    def test_first_step_has_no_predecessor(self) -> None:
        config = Pipeline("p").add(Step("load", "loader.general.x")).to_config()
        assert config.steps[0].after == []

    def test_a_step_with_no_explicit_after_chains_onto_the_previous_one(self) -> None:
        config = (
            Pipeline("p")
            .add(Step("load", "loader.general.x"))
            .add(Step("deid", "preprocessor.general.y"))
            .to_config()
        )
        assert config.steps[1].after == ["load"]

    def test_explicit_after_overrides_the_chaining_default(self) -> None:
        config = (
            Pipeline("p")
            .add(Step("load", "loader.general.x"))
            .add(Step("other", "loader.general.z", after=[]))
            .to_config()
        )
        assert config.steps[1].after == []

    def test_add_returns_self_for_further_chaining(self) -> None:
        pipeline = Pipeline("p")
        assert pipeline.add(Step("load", "loader.general.x")) is pipeline


class TestPipelineGuard:
    def test_a_single_string_at_becomes_a_one_element_list(self) -> None:
        config = (
            Pipeline("p")
            .add(Step("load", "loader.general.x"))
            .guard("guardrail.general.x", at="after:load")
            .to_config()
        )
        assert config.guardrails[0].at == ["after:load"]

    def test_a_list_at_is_kept_as_is(self) -> None:
        config = (
            Pipeline("p")
            .add(Step("load", "loader.general.x"))
            .guard("guardrail.general.x", at=["after:load", "after:chunk"])
            .to_config()
        )
        assert config.guardrails[0].at == ["after:load", "after:chunk"]

    def test_on_violation_defaults_to_block(self) -> None:
        config = (
            Pipeline("p")
            .add(Step("load", "loader.general.x"))
            .guard("guardrail.general.x", at="after:load")
            .to_config()
        )
        assert config.guardrails[0].on_violation == "block"

    def test_guard_returns_self_for_further_chaining(self) -> None:
        pipeline = Pipeline("p")
        assert pipeline.guard("guardrail.general.x", at="after:load") is pipeline


class TestToConfig:
    def test_produces_a_valid_pipelineconfig(self) -> None:
        config = (
            Pipeline("p", description="d")
            .add(Step("load", "loader.general.x", path="./notes"))
            .to_config()
        )
        assert isinstance(config, PipelineConfig)
        assert config.name == "p"
        assert config.description == "d"
        assert config.steps[0].params == {"path": "./notes"}

    def test_policy_and_provenance_pass_through(self) -> None:
        policy = PolicyConfig(allow_offsite_providers=True)
        provenance = ProvenanceConfig(manifest_dir="./custom_runs")
        config = (
            Pipeline("p", policy=policy, provenance=provenance)
            .add(Step("load", "loader.general.x"))
            .to_config()
        )
        assert config.policy.allow_offsite_providers is True
        assert config.provenance.manifest_dir == "./custom_runs"


class TestFromConfig:
    def test_round_trips_through_to_config(self) -> None:
        original = (
            Pipeline("p")
            .add(Step("load", "loader.general.x", path="./notes"))
            .add(Step("deid", "preprocessor.general.y"))
            .guard("guardrail.general.z", at="after:deid")
            .to_config()
        )
        rebuilt = Pipeline.from_config(original).to_config()
        assert rebuilt == original

    def test_a_multi_step_config_preserves_declared_after(self) -> None:
        """from_config must NOT re-infer chaining -- StepConfig.after is
        already explicit for every step it came from."""
        config = PipelineConfig(
            name="p",
            steps=[
                {"id": "a", "type": "loader.general.x"},
                {"id": "b", "type": "loader.general.y"},
            ],
        )
        rebuilt = Pipeline.from_config(config).to_config()
        assert rebuilt.steps[0].after == []
        assert rebuilt.steps[1].after == []


class TestFromYaml:
    def test_loads_a_real_yaml_file(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "pipeline.yaml"
        yaml_path.write_text(
            "name: p\nsteps:\n  - id: load\n    type: loader.general.x\n",
            encoding="utf-8",
        )
        pipeline = Pipeline.from_yaml(yaml_path)
        assert pipeline.to_config().name == "p"

    def test_missing_file_raises_configerror(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError):
            Pipeline.from_yaml(tmp_path / "does_not_exist.yaml")
