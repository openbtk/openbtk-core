"""Unit tests for openbtk.core.config.

0% coverage baseline: the contract suite never touches PipelineConfig at
all. Originally verified as a throwaway adversarial probe (real YAML
round-trip with env-var interpolation, the unset-variable failure path,
invalid-YAML and missing-file failure paths, validate_registry()'s registry
lookups and real DFS cycle detection including the diamond-DAG negative
control); converted to permanent tests here.

Deliberately self-contained: registers its own tiny loader and guardrail
below rather than depending on tests/contract/conftest.py having already
run in this process. A conftest.py has pytest-specific discovery semantics;
importing one as a plain module is not a pattern used elsewhere in this
codebase, and was not verified to survive being run from a directory
outside the repository (as CI's test-core job does).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from openbtk.core.base import BaseGuardrail, BaseLoader
from openbtk.core.config import (
    GuardrailConfig,
    PipelineConfig,
    PolicyConfig,
    ProvenanceConfig,
    StepConfig,
    ValidationIssue,
)
from openbtk.core.errors import ConfigError
from openbtk.core.registry import GUARDRAIL_REGISTRY, LOADER_REGISTRY
from openbtk.core.schemas import GuardrailResult, GuardrailSeverity

if TYPE_CHECKING:
    from pathlib import Path

_LOADER_KEY = "loader.general.config_test_probe"
_GUARDRAIL_KEY = "guardrail.general.config_test_probe"

if not LOADER_REGISTRY.is_registered(_LOADER_KEY):

    class _ProbeLoader(BaseLoader[Iterator[str], StepConfig]):
        """Matches tests/contract/conftest.py's ReferenceLoader shape
        exactly (Iterator[str] source, one record per item) -- registering
        into the real global LOADER_REGISTRY means the contract suite
        sweeps this up too (by design: no opt-out), and its generic checks
        assume every loader accepts an iterator of strings. Discovered when
        an earlier, always-empty stub here made the loader contract suite
        fail for a key it had never heard of until this file happened to
        import first -- a genuine test-order-dependent pollution bug, not a
        flaw in the contract suite itself."""

        def load(self, source: Iterator[str]) -> Iterator[StepConfig]:
            for i, line in enumerate(source):
                yield StepConfig(
                    id=f"probe-{i}", type=_LOADER_KEY, params={"line": line}
                )

    LOADER_REGISTRY.register(_LOADER_KEY)(_ProbeLoader)

if not GUARDRAIL_REGISTRY.is_registered(_GUARDRAIL_KEY):

    class _ProbeGuardrail(BaseGuardrail):
        def check(self, payload: Any) -> GuardrailResult:
            return GuardrailResult(
                passed=True,
                severity=GuardrailSeverity.INFO,
                guardrail_key=self.registry_key,
                message="ok",
            )

    GUARDRAIL_REGISTRY.register(_GUARDRAIL_KEY)(_ProbeGuardrail)


class TestDefaults:
    def test_policy_config_defaults_are_secure(self) -> None:
        policy = PolicyConfig()
        assert policy.allow_offsite_providers is False
        assert policy.fail_on_guardrail_block is True

    def test_provenance_config_default_manifest_dir(self) -> None:
        assert ProvenanceConfig().manifest_dir == "./runs"

    def test_guardrail_config_default_on_violation_is_block(self) -> None:
        gc = GuardrailConfig(type=_GUARDRAIL_KEY, at=["after:step"])
        assert gc.on_violation == "block"

    def test_step_config_default_params_and_after_are_empty(self) -> None:
        step = StepConfig(id="s1", type=_LOADER_KEY)
        assert step.params == {}
        assert step.after == []

    def test_pipeline_config_default_version_is_1(self) -> None:
        cfg = PipelineConfig(name="p", steps=[StepConfig(id="s1", type=_LOADER_KEY)])
        assert cfg.version == 1


class TestFrozenAndForbidExtra:
    def test_pipeline_config_is_frozen(self) -> None:
        cfg = PipelineConfig(name="p", steps=[StepConfig(id="s1", type=_LOADER_KEY)])
        with pytest.raises(ValidationError, match=r"(?i)frozen"):
            cfg.name = "mutated"  # type: ignore[misc]

    def test_pipeline_config_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError, match=r"(?i)extra"):
            PipelineConfig(
                name="p",
                bogus_field=1,  # type: ignore[call-arg]
                steps=[StepConfig(id="s1", type=_LOADER_KEY)],
            )

    def test_pipeline_config_requires_at_least_one_step(self) -> None:
        with pytest.raises(ValidationError):
            PipelineConfig(name="p", steps=[])


# A synthetic, unassigned placeholder value standing in for a real secret --
# proves env-var interpolation actually resolves a value, never a real
# credential.
_FAKE_INTERPOLATED_SECRET = "sk-real-secret-12345"  # pragma: allowlist secret


class TestEnvVarInterpolation:
    def test_interpolates_a_real_env_var(self, tmp_path: Path) -> None:
        os.environ["OPENBTK_TEST_PROBE_TOKEN"] = _FAKE_INTERPOLATED_SECRET
        try:
            yaml_file = tmp_path / "pipeline.yaml"
            yaml_file.write_text(
                "name: p\n"
                "steps:\n"
                "  - id: load\n"
                f"    type: {_LOADER_KEY}\n"
                "    params: {api_key: '${OPENBTK_TEST_PROBE_TOKEN}', plain: literal}\n"
            )
            cfg = PipelineConfig.from_yaml(yaml_file)
            assert cfg.steps[0].params["api_key"] == _FAKE_INTERPOLATED_SECRET
            assert cfg.steps[0].params["plain"] == "literal"
        finally:
            del os.environ["OPENBTK_TEST_PROBE_TOKEN"]

    def test_unset_env_var_raises_config_error_not_silent_fallback(
        self, tmp_path: Path
    ) -> None:
        yaml_file = tmp_path / "pipeline.yaml"
        yaml_file.write_text(
            "name: p\n"
            "steps:\n"
            "  - id: load\n"
            f"    type: {_LOADER_KEY}\n"
            "    params: {x: '${OPENBTK_TOTALLY_UNSET_VAR_XYZ}'}\n"
        )
        with pytest.raises(ConfigError, match="OPENBTK_TOTALLY_UNSET_VAR_XYZ"):
            PipelineConfig.from_yaml(yaml_file)

    def test_non_string_values_pass_through_untouched(self, tmp_path: Path) -> None:
        """_interpolate_env_vars recurses into dicts/lists looking for
        strings to interpolate; an int, bool, or None value must pass
        through unchanged rather than being stringified or dropped."""
        yaml_file = tmp_path / "pipeline.yaml"
        yaml_file.write_text(
            "name: p\n"
            "steps:\n"
            "  - id: load\n"
            f"    type: {_LOADER_KEY}\n"
            "    params: {max_tokens: 512, enabled: true, extra: null}\n"
        )
        cfg = PipelineConfig.from_yaml(yaml_file)
        params = cfg.steps[0].params
        assert params["max_tokens"] == 512
        assert params["enabled"] is True
        assert params["extra"] is None


class TestFromYamlFailurePaths:
    def test_invalid_yaml_raises_config_error(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("name: [unclosed")
        with pytest.raises(ConfigError):
            PipelineConfig.from_yaml(yaml_file)

    def test_missing_file_raises_config_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError):
            PipelineConfig.from_yaml(tmp_path / "does_not_exist.yaml")

    def test_schema_violation_raises_config_error(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "no_steps.yaml"
        yaml_file.write_text("name: p\nsteps: []\n")
        with pytest.raises(ConfigError):
            PipelineConfig.from_yaml(yaml_file)


class TestValidateRegistry:
    def test_clean_config_has_no_issues(self) -> None:
        cfg = PipelineConfig(name="ok", steps=[StepConfig(id="load", type=_LOADER_KEY)])
        assert cfg.validate_registry() == []

    def test_unknown_type_is_an_error(self) -> None:
        cfg = PipelineConfig(
            name="bad", steps=[StepConfig(id="load", type="loader.general.nope")]
        )
        issues = cfg.validate_registry()
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert all(isinstance(i, ValidationIssue) for i in issues)

    def test_unknown_category_is_an_error(self) -> None:
        cfg = PipelineConfig(
            name="bad", steps=[StepConfig(id="load", type="not_a_category.general.x")]
        )
        issues = cfg.validate_registry()
        assert len(issues) == 1

    def test_dangling_after_reference_is_an_error(self) -> None:
        cfg = PipelineConfig(
            name="bad",
            steps=[StepConfig(id="a", type=_LOADER_KEY, after=["nonexistent"])],
        )
        issues = cfg.validate_registry()
        assert any("nonexistent" in i.message for i in issues)

    def test_detects_a_real_two_node_cycle(self) -> None:
        cfg = PipelineConfig(
            name="cyclic",
            steps=[
                StepConfig(id="a", type=_LOADER_KEY, after=["b"]),
                StepConfig(id="b", type=_LOADER_KEY, after=["a"]),
            ],
        )
        issues = cfg.validate_registry()
        assert any("cycle" in i.message.lower() for i in issues)

    def test_diamond_dependency_is_not_a_false_positive_cycle(self) -> None:
        """The negative control: a DAG where a node is reached via two
        different paths (a->b->d, a->c->d) must NOT be flagged as a cycle.
        This is what catches a naive "visited twice = cycle" implementation."""
        cfg = PipelineConfig(
            name="diamond",
            steps=[
                StepConfig(id="a", type=_LOADER_KEY),
                StepConfig(id="b", type=_LOADER_KEY, after=["a"]),
                StepConfig(id="c", type=_LOADER_KEY, after=["a"]),
                StepConfig(id="d", type=_LOADER_KEY, after=["b", "c"]),
            ],
        )
        assert cfg.validate_registry() == []

    def test_unknown_guardrail_type_is_an_error(self) -> None:
        cfg = PipelineConfig(
            name="bad",
            steps=[StepConfig(id="a", type=_LOADER_KEY)],
            guardrails=[GuardrailConfig(type="guardrail.general.nope", at=["after:a"])],
        )
        issues = cfg.validate_registry()
        assert any("guardrail" in i.message.lower() for i in issues)

    def test_known_guardrail_type_has_no_issue(self) -> None:
        cfg = PipelineConfig(
            name="ok",
            steps=[StepConfig(id="a", type=_LOADER_KEY)],
            guardrails=[GuardrailConfig(type=_GUARDRAIL_KEY, at=["after:a"])],
        )
        assert cfg.validate_registry() == []


def test_json_schema_export_is_a_real_schema() -> None:
    schema = PipelineConfig.json_schema_export()
    assert schema["title"] == "PipelineConfig"
    assert "properties" in schema
    assert "steps" in schema["properties"]
