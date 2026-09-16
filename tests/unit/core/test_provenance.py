"""Unit tests for openbtk.core.provenance: ModelIdentity, ComponentProvenance,
DataDigest, GuardrailOutcome, StepProvenance, RunManifest.

Coverage gaps from the contract suite baseline: the floating-tag rejection
validator is never exercised there (no reference implementation constructs
a ModelIdentity), and ComponentProvenance's frozen/extra=forbid enforcement
is never adversarially tested. RunManifest and its parts have no contract
suite at all (they are not a Component base class) -- tested directly here.

TokenUsage's own tests live in tests/unit/core/test_schemas.py -- M5 task
5.1 moved its real definition to core.schemas (see that module's own
docstring for why); this file keeps only a check that the re-export
genuinely is the same class, not a second, drifting definition.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from openbtk.core import schemas as core_schemas
from openbtk.core.provenance import (
    ComponentProvenance,
    DataDigest,
    GuardrailOutcome,
    ModelIdentity,
    RunManifest,
    StepProvenance,
)
from openbtk.core.provenance import TokenUsage as ProvenanceTokenUsage


class TestModelIdentity:
    @pytest.mark.parametrize(
        "floating_tag",
        ["latest", "main", "master", "head", "stable", "LATEST", "  latest  "],
    )
    def test_rejects_every_floating_tag_case_and_whitespace_insensitively(
        self, floating_tag: str
    ) -> None:
        with pytest.raises(ValidationError, match="floating tag"):
            ModelIdentity(name="gpt-4o", revision=floating_tag, source="api")

    def test_accepts_a_real_commit_sha(self) -> None:
        fake_sha = "a1b2c3d4e5f67890abcdef"  # pragma: allowlist secret -- example SHA
        identity = ModelIdentity(
            name="meta-llama/Llama-3-8B", revision=fake_sha, source="huggingface"
        )
        assert identity.revision == fake_sha

    def test_is_frozen(self) -> None:
        identity = ModelIdentity(name="x", revision="abc123", source="api")
        with pytest.raises(ValidationError):
            identity.name = "mutated"  # type: ignore[misc]

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            ModelIdentity(
                name="x",
                revision="abc123",
                source="api",
                bogus_field=1,  # type: ignore[call-arg]
            )

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValidationError):
            ModelIdentity(name="", revision="abc123", source="api")


class TestComponentProvenance:
    def test_registry_key_accepts_empty_string(self) -> None:
        """The empty-string sentinel for "constructed outside the registry"
        (Component.registry_key's documented default) must be a valid
        ComponentProvenance -- this was a real bug, fixed once already this
        session, that a regression here would silently reintroduce."""
        cp = ComponentProvenance(
            registry_key="", class_name="Unregistered", package_version="0.1.0"
        )
        assert cp.registry_key == ""

    def test_config_defaults_to_empty_dict(self) -> None:
        cp = ComponentProvenance(
            registry_key="loader.general.x", class_name="X", package_version="0.1.0"
        )
        assert cp.config == {}

    def test_model_identity_defaults_to_none(self) -> None:
        cp = ComponentProvenance(
            registry_key="loader.general.x", class_name="X", package_version="0.1.0"
        )
        assert cp.model_identity is None

    def test_is_frozen(self) -> None:
        cp = ComponentProvenance(
            registry_key="loader.general.x", class_name="X", package_version="0.1.0"
        )
        with pytest.raises(ValidationError):
            cp.class_name = "mutated"  # type: ignore[misc]

    def test_serialises_to_valid_json_with_a_nested_model_identity(self) -> None:
        cp = ComponentProvenance(
            registry_key="embedding.general.x",
            class_name="X",
            package_version="0.1.0",
            config={"device": "cpu"},
            model_identity=ModelIdentity(name="m", revision="abc123", source="api"),
        )
        dumped = cp.model_dump_json()
        restored = ComponentProvenance.model_validate_json(dumped)
        assert restored == cp


class TestDataDigest:
    def test_sha256_defaults_to_none(self) -> None:
        assert DataDigest(uri="./notes", record_count=0).sha256 is None

    def test_rejects_negative_record_count(self) -> None:
        with pytest.raises(ValidationError):
            DataDigest(uri="./notes", record_count=-1)

    def test_rejects_empty_uri(self) -> None:
        with pytest.raises(ValidationError):
            DataDigest(uri="", record_count=0)

    def test_is_frozen(self) -> None:
        digest = DataDigest(uri="./notes", record_count=3)
        with pytest.raises(ValidationError):
            digest.record_count = 4  # type: ignore[misc]


class TestTokenUsageReExport:
    def test_provenance_tokenusage_is_the_same_class_as_schemas_tokenusage(
        self,
    ) -> None:
        """A genuine re-export, not a second, drifting definition -- see
        this module's own docstring for the import-cycle reason it moved."""
        assert ProvenanceTokenUsage is core_schemas.TokenUsage


class TestGuardrailOutcome:
    def test_blocked_and_warned_default_to_zero(self) -> None:
        outcome = GuardrailOutcome(
            guardrail_key="guardrail.general.phi_leakage",
            at="after:deid",
            checked_count=10,
        )
        assert (outcome.blocked_count, outcome.warned_count) == (0, 0)

    def test_sample_messages_default_to_empty_list(self) -> None:
        outcome = GuardrailOutcome(
            guardrail_key="guardrail.general.phi_leakage",
            at="after:deid",
            checked_count=0,
        )
        assert outcome.sample_messages == []

    def test_rejects_negative_checked_count(self) -> None:
        with pytest.raises(ValidationError):
            GuardrailOutcome(
                guardrail_key="guardrail.general.phi_leakage",
                at="after:deid",
                checked_count=-1,
            )


class TestStepProvenance:
    def _component(self) -> ComponentProvenance:
        return ComponentProvenance(
            registry_key="loader.clinical_text.plain_text",
            class_name="PlainTextLoader",
            package_version="0.1.0",
        )

    def test_error_defaults_to_none(self) -> None:
        sp = StepProvenance(
            step_id="load",
            component=self._component(),
            records_in=0,
            records_out=3,
            status="success",
        )
        assert sp.error is None

    def test_rejects_status_outside_success_or_failed(self) -> None:
        with pytest.raises(ValidationError):
            StepProvenance(
                step_id="load",
                component=self._component(),
                records_in=0,
                records_out=0,
                status="partial",
            )

    def test_rejects_negative_counts(self) -> None:
        with pytest.raises(ValidationError):
            StepProvenance(
                step_id="load",
                component=self._component(),
                records_in=-1,
                records_out=0,
                status="success",
            )


class TestRunManifest:
    def _manifest(self, **overrides: object) -> RunManifest:
        defaults: dict[str, object] = {
            "run_id": "a1b2c3",
            "status": "success",
            "config": {"name": "probe", "steps": []},
            "started_at": datetime(2026, 1, 1, tzinfo=UTC),
        }
        defaults.update(overrides)
        return RunManifest(**defaults)

    def test_manifest_version_defaults_to_1_0_0(self) -> None:
        assert self._manifest().manifest_version == "1.0.0"

    def test_ended_at_defaults_to_none(self) -> None:
        assert self._manifest().ended_at is None

    def test_steps_and_digests_and_outcomes_default_to_empty(self) -> None:
        manifest = self._manifest()
        assert manifest.steps == []
        assert manifest.input_digests == []
        assert manifest.guardrail_outcomes == []

    def test_token_usage_defaults_to_none(self) -> None:
        assert self._manifest().token_usage is None

    def test_rejects_status_outside_the_three_known_values(self) -> None:
        with pytest.raises(ValidationError):
            self._manifest(status="cancelled")

    def test_is_frozen(self) -> None:
        manifest = self._manifest()
        with pytest.raises(ValidationError):
            manifest.status = "failed"  # type: ignore[misc]

    def test_round_trips_through_json_with_nested_steps(self) -> None:
        manifest = self._manifest(
            steps=[
                StepProvenance(
                    step_id="load",
                    component=ComponentProvenance(
                        registry_key="loader.general.x",
                        class_name="X",
                        package_version="0.1.0",
                    ),
                    records_in=0,
                    records_out=3,
                    status="success",
                )
            ]
        )
        restored = RunManifest.model_validate_json(manifest.model_dump_json())
        assert restored == manifest

    def test_config_never_needs_to_be_a_pipelineconfig_instance(self) -> None:
        """RunManifest.config is a plain, JSON-safe snapshot -- see this
        module's own docstring for why (a layering cycle, not a stylistic
        choice)."""
        manifest = self._manifest(config={"anything": ["json", "safe", 1, None]})
        assert manifest.config == {"anything": ["json", "safe", 1, None]}
