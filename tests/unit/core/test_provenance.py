"""Unit tests for openbtk.core.provenance (ModelIdentity, ComponentProvenance).

Coverage gaps from the contract suite baseline: the floating-tag rejection
validator is never exercised there (no reference implementation constructs
a ModelIdentity), and ComponentProvenance's frozen/extra=forbid enforcement
is never adversarially tested.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from openbtk.core.provenance import ComponentProvenance, ModelIdentity


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
        identity = ModelIdentity(
            name="meta-llama/Llama-3-8B",
            revision="a1b2c3d4e5f67890abcdef",
            source="huggingface",
        )
        assert identity.revision == "a1b2c3d4e5f67890abcdef"

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
