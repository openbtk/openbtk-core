"""
openbtk pipeline and component configuration system.

Pipelines are defined declaratively as PipelineConfig objects (from YAML,
JSON, or Python dicts). Each step references a registry key and optional
constructor parameters.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from openbtk.core.errors import ConfigError


class StepConfig(BaseModel):
    """Configuration for a single pipeline step."""

    model_config = {"frozen": True}

    type: str = Field(
        ...,
        description=(
            "Registry key identifying the component class. "
            "Format: '<category>.<name>' e.g. 'clinical_text.section_aware_chunker'"
        ),
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Constructor keyword arguments passed to the component.",
    )

    @model_validator(mode="after")
    def validate_type_format(self) -> "StepConfig":
        if "." not in self.type:
            raise ValueError(
                f"StepConfig.type must be a dotted registry key "
                f"(e.g. 'loader.mimic_notes'), got: '{self.type}'"
            )
        return self


class PipelineConfig(BaseModel):
    """Declarative configuration for a full openbtk pipeline."""

    model_config = {"frozen": True}

    name: str = Field(..., description="Human-readable pipeline name.")
    version: str = Field("0.1.0", description="Pipeline definition version.")
    description: str | None = Field(None, description="Optional pipeline description.")
    steps: list[StepConfig] = Field(
        ...,
        min_length=1,
        description="Ordered list of pipeline steps.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary pipeline-level metadata (author, tags, etc.).",
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load a PipelineConfig from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Parsed and validated PipelineConfig.

        Raises:
            ConfigError: If the file cannot be read or the YAML is invalid.
        """
        p = Path(path)
        if not p.exists():
            raise ConfigError(
                f"Pipeline config file not found: {p}",
                context={"path": str(p)},
            )
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise ConfigError(
                f"Invalid YAML in pipeline config: {p}",
                context={"path": str(p)},
            ) from e
        return cls._from_raw(raw, source=str(p))

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        """Load a PipelineConfig from a JSON file.

        Args:
            path: Path to the JSON configuration file.

        Returns:
            Parsed and validated PipelineConfig.

        Raises:
            ConfigError: If the file cannot be read or the JSON is invalid.
        """
        p = Path(path)
        if not p.exists():
            raise ConfigError(
                f"Pipeline config file not found: {p}",
                context={"path": str(p)},
            )
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ConfigError(
                f"Invalid JSON in pipeline config: {p}",
                context={"path": str(p)},
            ) from e
        return cls._from_raw(raw, source=str(p))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        """Construct a PipelineConfig from a plain dict.

        Args:
            data: Dict matching PipelineConfig schema.

        Returns:
            Validated PipelineConfig.

        Raises:
            ConfigError: On Pydantic validation failure.
        """
        return cls._from_raw(data, source="dict")

    @classmethod
    def _from_raw(cls, raw: Any, source: str) -> "PipelineConfig":
        if not isinstance(raw, dict):
            raise ConfigError(
                f"Pipeline config must be a mapping, got {type(raw).__name__}",
                context={"source": source},
            )
        try:
            return cls.model_validate(raw)
        except Exception as e:
            raise ConfigError(
                f"Pipeline config validation failed from {source}: {e}",
                context={"source": source},
            ) from e

    def to_yaml(self, path: str | Path) -> None:
        """Serialize this config to a YAML file."""
        Path(path).write_text(
            yaml.dump(self.model_dump(), default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )

    def to_json(self, path: str | Path) -> None:
        """Serialize this config to a JSON file."""
        Path(path).write_text(
            self.model_dump_json(indent=2),
            encoding="utf-8",
        )


class ProviderConfig(BaseModel):
    """Generic provider configuration (LLM, embedding, vector store)."""

    model_config = {"frozen": True}

    provider_type: str = Field(
        ...,
        description="Registry key for the provider, e.g. 'embeddings.pubmedbert'.",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Constructor parameters for the provider.",
    )


class LoggingConfig(BaseModel):
    """Logging configuration for openbtk."""

    model_config = {"frozen": True}

    level: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    fmt: str = Field("json", pattern="^(json|console)$")
    redact_keys: list[str] = Field(
        default_factory=list,
        description="Additional field keys to redact from logs beyond defaults.",
    )


class openbtkConfig(BaseModel):
    """Top-level application configuration.

    Combines logging, provider defaults, and optional named pipelines.
    """

    model_config = {"frozen": True}

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    default_llm: ProviderConfig | None = None
    default_embedding: ProviderConfig | None = None
    default_vectorstore: ProviderConfig | None = None
    pipelines: dict[str, PipelineConfig] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "openbtkConfig":
        """Load top-level config from YAML."""
        p = Path(path)
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        except (FileNotFoundError, yaml.YAMLError) as e:
            raise ConfigError(f"Cannot load config: {p}", context={"path": str(p)}) from e
        try:
            return cls.model_validate(raw or {})
        except Exception as e:
            raise ConfigError(f"Config validation failed: {e}") from e
