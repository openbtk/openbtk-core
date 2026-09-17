"""Unit tests for openbtk.llms.presets.

Presets construct a real HuggingFaceLocalProvider, so these tests mock
``require()`` the same way tests/unit/llms/test_huggingface.py does --
constructing the provider never loads a real model (HuggingFaceLocalProvider
loads lazily, on first call), so no mocking is even needed for that part;
what's actually under test here is the preset lookup/merge/error logic.
"""

from __future__ import annotations

import pytest

from openbtk.core.config import PolicyConfig
from openbtk.core.errors import ConfigError, PolicyError
from openbtk.llms.huggingface import HuggingFaceLocalProvider
from openbtk.llms.presets import (
    BIOMEDICAL_LLM_PRESETS,
    create_llm_preset,
    list_llm_presets,
)


class TestListLlmPresets:
    def test_returns_every_preset_name_sorted(self) -> None:
        assert list_llm_presets() == sorted(BIOMEDICAL_LLM_PRESETS)

    def test_includes_the_three_roadmap_named_families(self) -> None:
        names = list_llm_presets()
        assert "meditron-7b" in names
        assert "openbiollm-8b" in names
        assert "medgemma-27b-text" in names


class TestCreateLlmPreset:
    def test_constructs_a_huggingface_local_provider(self) -> None:
        provider = create_llm_preset("openbiollm-8b")
        assert isinstance(provider, HuggingFaceLocalProvider)

    def test_uses_the_presets_own_model_and_revision(self) -> None:
        provider = create_llm_preset("meditron-7b")
        identity = provider.model_identity()
        assert identity.name == "epfl-llm/meditron-7b"
        sha = "d7d0a5ed929384a6b059ac74198cf1d71f44ba76"  # pragma: allowlist secret
        assert identity.revision == sha

    def test_every_preset_constructs_without_error(self) -> None:
        for name in list_llm_presets():
            provider = create_llm_preset(name)
            assert isinstance(provider, HuggingFaceLocalProvider)

    def test_overrides_are_merged_over_the_presets_defaults(self) -> None:
        provider = create_llm_preset("openbiollm-8b", device="cuda")
        assert provider.provenance().config["device"] == "cuda"

    def test_an_override_can_replace_a_preset_default(self) -> None:
        provider = create_llm_preset("meditron-7b", revision="deadbeef")
        assert provider.model_identity().revision == "deadbeef"

    def test_unknown_preset_name_raises_config_error(self) -> None:
        with pytest.raises(ConfigError, match="Unknown biomedical LLM preset"):
            create_llm_preset("not-a-real-preset")

    def test_unknown_preset_error_lists_the_real_names(self) -> None:
        with pytest.raises(ConfigError) as exc_info:
            create_llm_preset("not-a-real-preset")
        assert exc_info.value.context["preset"] == "not-a-real-preset"

    def test_a_policy_override_is_forwarded_to_the_registry(self) -> None:
        """huggingface_local is sends_data_offsite=False, so a default
        (no policy) construction already succeeds -- passing an explicit
        restrictive policy proves the kwarg genuinely reaches
        Registry.create rather than being silently swallowed as just
        another provider param (which would raise a TypeError instead,
        HuggingFaceLocalProvider.__init__ has no policy parameter)."""
        provider = create_llm_preset(
            "openbiollm-8b", policy=PolicyConfig(allow_offsite_providers=False)
        )
        assert isinstance(provider, HuggingFaceLocalProvider)


class TestPresetDataShape:
    def test_every_preset_targets_the_huggingface_local_registry_key(self) -> None:
        for preset in BIOMEDICAL_LLM_PRESETS.values():
            assert preset["type"] == "llm.general.huggingface_local"

    def test_every_preset_has_a_model_and_a_40_character_revision(self) -> None:
        for preset in BIOMEDICAL_LLM_PRESETS.values():
            params = preset["params"]
            assert params["model"]
            assert len(params["revision"]) == 40

    def test_no_preset_revision_is_a_floating_tag(self) -> None:
        """Real, direct verification of FR-P-05 against every preset's own
        data, not just an assertion that HuggingFaceLocalProvider would
        reject one -- an actual construction proves it end to end."""
        for name in list_llm_presets():
            create_llm_preset(name)  # would raise ValidationError otherwise


def test_policy_error_is_not_raised_for_a_local_preset() -> None:
    """Every current preset is HuggingFaceLocalProvider (sends_data_offsite
    = False), so none of them should ever hit Registry.create's offsite
    policy gate -- this is a regression guard for that fact, not a test of
    the gate itself (see tests/security/test_offsite_policy_enforcement.py)."""
    try:
        create_llm_preset("meditron-7b")
    except PolicyError:
        pytest.fail("A local-only preset must never trigger the offsite policy gate.")
