"""Unit tests for openbtk.core.config."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from openbtk.core.config import PipelineConfig, StepConfig, LoggingConfig
from openbtk.core.errors import ConfigError


def test_step_config_requires_dotted_type() -> None:
    with pytest.raises(ValidationError, match="dotted registry key"):
        StepConfig(type="not_dotted")


def test_step_config_valid() -> None:
    step = StepConfig(type="loader.clinical_text.plain_text", params={"a": 1})
    assert step.type == "loader.clinical_text.plain_text"
    assert step.params == {"a": 1}


def test_step_config_default_params_empty() -> None:
    step = StepConfig(type="loader.clinical_text.plain_text")
    assert step.params == {}


def test_pipeline_config_requires_at_least_one_step() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig(name="empty-pipeline", steps=[])


def test_pipeline_config_from_dict() -> None:
    cfg = PipelineConfig.from_dict({
        "name": "test-pipeline",
        "steps": [
            {"type": "loader.clinical_text.plain_text", "params": {}},
            {"type": "chunker.clinical_text.section_aware", "params": {"max_tokens": 256}},
        ],
    })
    assert cfg.name == "test-pipeline"
    assert len(cfg.steps) == 2
    assert cfg.steps[1].params["max_tokens"] == 256


def test_pipeline_config_from_dict_missing_required_field_raises() -> None:
    with pytest.raises(ConfigError):
        PipelineConfig.from_dict({"steps": [{"type": "loader.x"}]})  # missing name


def test_pipeline_config_yaml_roundtrip(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        name="roundtrip-test",
        description="A test pipeline",
        steps=[
            StepConfig(type="loader.clinical_text.plain_text", params={"note_type": "H&P"}),
            StepConfig(type="chunker.clinical_text.fixed_token", params={"max_tokens": 128}),
        ],
    )
    yaml_path = tmp_path / "pipeline.yaml"
    cfg.to_yaml(yaml_path)

    loaded = PipelineConfig.from_yaml(yaml_path)
    assert loaded.name == cfg.name
    assert loaded.description == cfg.description
    assert len(loaded.steps) == len(cfg.steps)
    assert loaded.steps[0].type == cfg.steps[0].type
    assert loaded.steps[1].params["max_tokens"] == 128


def test_pipeline_config_from_yaml_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(ConfigError, match="not found"):
        PipelineConfig.from_yaml(missing)


def test_pipeline_config_from_yaml_invalid_yaml_raises(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("name: [unclosed list\n  steps: oops")

    with pytest.raises(ConfigError):
        PipelineConfig.from_yaml(bad_yaml)


def test_pipeline_config_json_roundtrip(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        name="json-test",
        steps=[StepConfig(type="loader.clinical_text.plain_text")],
    )
    json_path = tmp_path / "pipeline.json"
    cfg.to_json(json_path)

    loaded = PipelineConfig.from_json(json_path)
    assert loaded.name == "json-test"


def test_pipeline_config_is_frozen() -> None:
    cfg = PipelineConfig(
        name="frozen-test",
        steps=[StepConfig(type="loader.clinical_text.plain_text")],
    )
    with pytest.raises(ValidationError):
        cfg.name = "changed"  # type: ignore[misc]


def test_logging_config_defaults() -> None:
    cfg = LoggingConfig()
    assert cfg.level == "INFO"
    assert cfg.fmt == "json"
    assert cfg.redact_keys == []


def test_logging_config_invalid_level_raises() -> None:
    with pytest.raises(ValidationError):
        LoggingConfig(level="NOT_A_LEVEL")
