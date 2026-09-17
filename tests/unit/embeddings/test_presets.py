"""Unit tests for openbtk.embeddings.presets.

Presets construct a real HuggingFaceEmbeddingProvider, which does no I/O
in __init__ or in its dimension/model_identity/provenance methods, so no
mocking is needed for what's under test here: the preset lookup/merge/
error logic itself (mirrors tests/unit/llms/test_presets.py).
"""

from __future__ import annotations

import pytest

from openbtk.core.config import PolicyConfig
from openbtk.core.errors import ConfigError, PolicyError
from openbtk.embeddings.huggingface import HuggingFaceEmbeddingProvider
from openbtk.embeddings.presets import (
    BIOMEDICAL_EMBEDDING_PRESETS,
    create_embedding_preset,
    list_embedding_presets,
)


class TestListEmbeddingPresets:
    def test_returns_every_preset_name_sorted(self) -> None:
        assert list_embedding_presets() == sorted(BIOMEDICAL_EMBEDDING_PRESETS)

    def test_includes_the_five_roadmap_named_families(self) -> None:
        names = list_embedding_presets()
        assert "pubmedbert" in names
        assert "biobert" in names
        assert "clinicalbert" in names
        assert "sapbert" in names
        # MedCPT is a dual encoder -- named explicitly, see the module docstring.
        assert "medcpt-query" in names
        assert "medcpt-article" in names


class TestCreateEmbeddingPreset:
    def test_constructs_a_huggingface_embedding_provider(self) -> None:
        provider = create_embedding_preset("biobert")
        assert isinstance(provider, HuggingFaceEmbeddingProvider)

    def test_uses_the_presets_own_model_and_revision(self) -> None:
        provider = create_embedding_preset("clinicalbert")
        identity = provider.model_identity()
        assert identity.name == "emilyalsentzer/Bio_ClinicalBERT"
        assert len(identity.revision) == 40

    def test_every_preset_constructs_without_error(self) -> None:
        for name in list_embedding_presets():
            provider = create_embedding_preset(name)
            assert isinstance(provider, HuggingFaceEmbeddingProvider)
            assert provider.dimension == 768

    def test_overrides_are_merged_over_the_presets_defaults(self) -> None:
        provider = create_embedding_preset("sapbert", device="cuda")
        assert provider.provenance().config["device"] == "cuda"

    def test_an_override_can_replace_a_preset_default(self) -> None:
        provider = create_embedding_preset("biobert", pooling="cls")
        assert provider.provenance().config["pooling"] == "cls"

    def test_sapbert_and_medcpt_default_to_cls_pooling(self) -> None:
        for name in ("sapbert", "medcpt-query", "medcpt-article"):
            provider = create_embedding_preset(name)
            assert provider.provenance().config["pooling"] == "cls"

    def test_pubmedbert_biobert_clinicalbert_default_to_mean_pooling(self) -> None:
        for name in ("pubmedbert", "biobert", "clinicalbert"):
            provider = create_embedding_preset(name)
            assert provider.provenance().config["pooling"] == "mean"

    def test_unknown_preset_name_raises_config_error(self) -> None:
        with pytest.raises(ConfigError, match="Unknown biomedical embedding preset"):
            create_embedding_preset("not-a-real-preset")

    def test_unknown_preset_error_lists_the_real_names(self) -> None:
        with pytest.raises(ConfigError) as exc_info:
            create_embedding_preset("not-a-real-preset")
        assert exc_info.value.context["preset"] == "not-a-real-preset"

    def test_a_policy_override_is_forwarded_to_the_registry(self) -> None:
        provider = create_embedding_preset(
            "biobert", policy=PolicyConfig(allow_offsite_providers=False)
        )
        assert isinstance(provider, HuggingFaceEmbeddingProvider)


class TestPresetDataShape:
    def test_every_preset_targets_the_huggingface_registry_key(self) -> None:
        for preset in BIOMEDICAL_EMBEDDING_PRESETS.values():
            assert preset["type"] == "embedding.general.huggingface"

    def test_every_preset_has_a_model_a_40_char_revision_and_a_dimension(
        self,
    ) -> None:
        for preset in BIOMEDICAL_EMBEDDING_PRESETS.values():
            params = preset["params"]
            assert params["model"]
            assert len(params["revision"]) == 40
            assert params["dimension"] > 0

    def test_medcpt_query_and_article_use_different_repositories(self) -> None:
        query = BIOMEDICAL_EMBEDDING_PRESETS["medcpt-query"]["params"]
        article = BIOMEDICAL_EMBEDDING_PRESETS["medcpt-article"]["params"]
        assert query["model"] != article["model"]
        assert "Query" in query["model"]
        assert "Article" in article["model"]


def test_policy_error_is_not_raised_for_a_local_preset() -> None:
    """Every current embedding preset is HuggingFaceEmbeddingProvider
    (sends_data_offsite=False) -- regression guard, not a test of the gate
    itself (see tests/security/test_offsite_policy_enforcement.py)."""
    try:
        create_embedding_preset("pubmedbert")
    except PolicyError:
        pytest.fail("A local-only preset must never trigger the offsite policy gate.")
