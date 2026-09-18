"""Unit tests for openbtk.embeddings.openai.OpenAIEmbeddingProvider.

Same rationale as tests/unit/llms/test_openai.py: the ``openai`` package
is optional, so the SDK surface is mocked directly.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from openbtk.core.errors import (
    AuthenticationError,
    ConfigError,
    ProviderError,
    RateLimitError,
)
from openbtk.embeddings.openai import OpenAIEmbeddingProvider


class _FakeRateLimitError(Exception):
    pass


class _FakeAuthenticationError(Exception):
    pass


class _FakeAPIError(Exception):
    pass


def _fake_openai_module(client: Any) -> Any:
    module = MagicMock()
    module.OpenAI = MagicMock(return_value=client)
    module.RateLimitError = _FakeRateLimitError
    module.AuthenticationError = _FakeAuthenticationError
    module.APIError = _FakeAPIError
    return module


def _embeddings_response(vectors: list[list[float]]) -> Any:
    response = MagicMock()
    response.data = [MagicMock(embedding=v) for v in vectors]
    return response


@pytest.fixture
def fake_client() -> Any:
    return MagicMock()


@pytest.fixture(autouse=True)
def _patch_require(monkeypatch: pytest.MonkeyPatch, fake_client: Any) -> None:
    monkeypatch.setattr(
        "openbtk.embeddings.openai.require",
        lambda module, extra: _fake_openai_module(fake_client),
    )


class TestEmbed:
    def test_returns_a_float32_array_shaped_by_batch_and_dimension(
        self, fake_client: Any
    ) -> None:
        fake_client.embeddings.create.return_value = _embeddings_response(
            [[0.1, 0.2], [0.3, 0.4]]
        )
        provider = OpenAIEmbeddingProvider(dimension=2)
        vectors = provider.embed(["a", "b"])
        assert vectors.shape == (2, 2)
        assert vectors.dtype == np.float32
        np.testing.assert_allclose(vectors, [[0.1, 0.2], [0.3, 0.4]], atol=1e-6)

    def test_forwards_the_configured_model_and_input_texts(
        self, fake_client: Any
    ) -> None:
        fake_client.embeddings.create.return_value = _embeddings_response([[0.0]])
        provider = OpenAIEmbeddingProvider(model="text-embedding-3-large", dimension=1)
        provider.embed(["hello"])
        _, call_kwargs = fake_client.embeddings.create.call_args
        assert call_kwargs["model"] == "text-embedding-3-large"
        assert call_kwargs["input"] == ["hello"]


class TestKnownDimensions:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("text-embedding-3-small", 1536),
            ("text-embedding-3-large", 3072),
            ("text-embedding-ada-002", 1536),
        ],
    )
    def test_known_models_default_their_documented_dimension(
        self, model: str, expected: int
    ) -> None:
        provider = OpenAIEmbeddingProvider(model=model)
        assert provider.dimension == expected

    def test_an_unknown_model_without_an_explicit_dimension_raises(self) -> None:
        with pytest.raises(ConfigError, match="Unknown OpenAI embedding model"):
            OpenAIEmbeddingProvider(model="some-future-model")

    def test_an_unknown_model_with_an_explicit_dimension_is_accepted(self) -> None:
        provider = OpenAIEmbeddingProvider(model="some-future-model", dimension=42)
        assert provider.dimension == 42

    def test_an_explicit_dimension_overrides_the_known_default(self) -> None:
        provider = OpenAIEmbeddingProvider(
            model="text-embedding-3-small", dimension=256
        )
        assert provider.dimension == 256


class TestErrorTranslation:
    def test_persistent_rate_limit_error_raises_openbtks_rate_limit_error(
        self, fake_client: Any
    ) -> None:
        fake_client.embeddings.create.side_effect = _FakeRateLimitError("nope")
        provider = OpenAIEmbeddingProvider(max_retries=1)
        with pytest.raises(RateLimitError):
            provider.embed(["hi"])

    def test_authentication_error_is_translated_and_not_retried(
        self, fake_client: Any
    ) -> None:
        calls = []

        def always_fails(*args: object, **kwargs: object) -> Any:
            calls.append(1)
            raise _FakeAuthenticationError("bad key")

        fake_client.embeddings.create.side_effect = always_fails
        provider = OpenAIEmbeddingProvider(max_retries=5)
        with pytest.raises(AuthenticationError):
            provider.embed(["hi"])
        assert len(calls) == 1

    def test_generic_api_error_is_translated_to_provider_error(
        self, fake_client: Any
    ) -> None:
        fake_client.embeddings.create.side_effect = _FakeAPIError("boom")
        provider = OpenAIEmbeddingProvider(max_retries=1)
        with pytest.raises(ProviderError):
            provider.embed(["hi"])


class TestModelIdentityAndProvenance:
    def test_model_identity_uses_the_configured_model(self) -> None:
        provider = OpenAIEmbeddingProvider(model="text-embedding-3-large")
        identity = provider.model_identity()
        assert identity.name == "text-embedding-3-large"
        assert identity.revision == "text-embedding-3-large"
        assert identity.source == "api"

    def test_provenance_config_never_includes_the_api_key(self) -> None:
        provider = OpenAIEmbeddingProvider(api_key="should-never-appear")
        dumped = provider.provenance().model_dump_json()
        assert "should-never-appear" not in dumped

    def test_provenance_records_model_and_dimension(self) -> None:
        provider = OpenAIEmbeddingProvider(model="text-embedding-3-large")
        provenance = provider.provenance()
        assert provenance.config == {
            "model": "text-embedding-3-large",
            "dimension": 3072,
        }


class TestClientLaziness:
    def test_constructing_the_provider_does_not_build_the_sdk_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []
        monkeypatch.setattr(
            "openbtk.embeddings.openai.require",
            lambda module, extra: calls.append(1) or _fake_openai_module(MagicMock()),
        )
        OpenAIEmbeddingProvider()
        assert calls == []

    def test_the_client_is_reused_across_calls(self, fake_client: Any) -> None:
        fake_client.embeddings.create.return_value = _embeddings_response([[0.0]])
        provider = OpenAIEmbeddingProvider(dimension=1)
        provider.embed(["one"])
        provider.embed(["two"])
        assert fake_client.embeddings.create.call_count == 2


class TestDeclaredAttributes:
    def test_sends_data_offsite_is_true(self) -> None:
        assert OpenAIEmbeddingProvider.sends_data_offsite is True

    def test_registered_under_the_expected_key(self) -> None:
        assert OpenAIEmbeddingProvider.registry_key == "embedding.general.openai"
