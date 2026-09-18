"""Unit tests for openbtk.embeddings.huggingface.HuggingFaceEmbeddingProvider.

Neither ``torch`` nor ``transformers`` needs to be installed: both are
mocked, with a small numpy-backed ``_FakeTensor`` standing in for a real
``torch.Tensor`` -- just enough surface (``unsqueeze``/``to``/``clamp``/
arithmetic/``detach``/``cpu``/``numpy``) for the provider's actual pooling
math to run unmodified against real numpy arithmetic underneath, the same
rationale as tests/unit/llms/test_huggingface.py's ``_FakeTokenIds``.
"""

from __future__ import annotations

import contextlib
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from openbtk.core.errors import ProviderError
from openbtk.embeddings.huggingface import HuggingFaceEmbeddingProvider


class _FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self.array = np.asarray(array)

    def __getitem__(self, item: Any) -> _FakeTensor:
        return _FakeTensor(self.array[item])

    def unsqueeze(self, dim: int) -> _FakeTensor:
        return _FakeTensor(np.expand_dims(self.array, dim))

    def to(self, dtype_or_device: Any) -> _FakeTensor:
        return self

    def __mul__(self, other: _FakeTensor) -> _FakeTensor:
        return _FakeTensor(self.array * other.array)

    def sum(self, dim: int) -> _FakeTensor:
        return _FakeTensor(self.array.sum(axis=dim))

    def clamp(self, min: float) -> _FakeTensor:
        return _FakeTensor(np.clip(self.array, min, None))

    def __truediv__(self, other: _FakeTensor) -> _FakeTensor:
        return _FakeTensor(self.array / other.array)

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.array

    @property
    def dtype(self) -> Any:
        return self.array.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape


class _FakeBatchEncoding(dict[str, Any]):
    def to(self, device: str) -> _FakeBatchEncoding:
        return self


# batch of 2, seq_len 3, hidden 4. Example 0's third token is masked out
# (padding); example 1's are all real -- deliberately different so mean
# vs. CLS pooling produce distinguishable, hand-computable results.
_HIDDEN_STATE = np.array(
    [
        [[1, 1, 1, 1], [3, 3, 3, 3], [99, 99, 99, 99]],
        [[2, 2, 2, 2], [4, 4, 4, 4], [6, 6, 6, 6]],
    ],
    dtype=np.float32,
)
_ATTENTION_MASK = np.array([[1, 1, 0], [1, 1, 1]], dtype=np.float32)


def _fake_torch_module() -> Any:
    module = MagicMock()
    module.no_grad = contextlib.nullcontext
    return module


def _fake_transformers_module(tokenizer: Any, model: Any) -> Any:
    module = MagicMock()
    module.AutoTokenizer.from_pretrained = MagicMock(return_value=tokenizer)
    module.AutoModel.from_pretrained = MagicMock(return_value=model)
    return module


@pytest.fixture
def fake_tokenizer() -> Any:
    tokenizer = MagicMock()
    tokenizer.side_effect = (
        lambda texts, padding, truncation, max_length, return_tensors: (
            _FakeBatchEncoding(
                input_ids=_FakeTensor(np.zeros((len(texts), 3))),
                attention_mask=_FakeTensor(_ATTENTION_MASK.copy()),
            )
        )
    )
    return tokenizer


@pytest.fixture
def fake_model() -> Any:
    model = MagicMock()
    model.to = MagicMock(return_value=model)
    model.side_effect = lambda **kwargs: MagicMock(
        last_hidden_state=_FakeTensor(_HIDDEN_STATE.copy())
    )
    return model


@pytest.fixture(autouse=True)
def _patch_require(
    monkeypatch: pytest.MonkeyPatch, fake_tokenizer: Any, fake_model: Any
) -> None:
    def fake_require(module: str, extra: str) -> Any:
        if module == "torch":
            return _fake_torch_module()
        if module == "transformers":
            return _fake_transformers_module(fake_tokenizer, fake_model)
        raise AssertionError(f"unexpected require({module!r})")

    monkeypatch.setattr("openbtk.embeddings.huggingface.require", fake_require)


def _provider(**kwargs: Any) -> HuggingFaceEmbeddingProvider:
    kwargs.setdefault("model", "tiny/test-model")
    kwargs.setdefault("revision", "abc123def456")  # pragma: allowlist secret
    kwargs.setdefault("dimension", 4)
    return HuggingFaceEmbeddingProvider(**kwargs)


class TestEmbedMeanPooling:
    def test_masks_out_padding_tokens(self) -> None:
        provider = _provider(pooling="mean")
        vectors = provider.embed(["a", "b"])
        expected = np.array([[2, 2, 2, 2], [4, 4, 4, 4]], dtype=np.float32)
        np.testing.assert_array_equal(vectors, expected)

    def test_returns_float32(self) -> None:
        provider = _provider(pooling="mean")
        vectors = provider.embed(["a", "b"])
        assert vectors.dtype == np.float32

    def test_shape_matches_batch_size_and_dimension(self) -> None:
        provider = _provider(pooling="mean")
        vectors = provider.embed(["a", "b"])
        assert vectors.shape == (2, 4)


class TestEmbedClsPooling:
    def test_uses_only_the_first_token(self) -> None:
        provider = _provider(pooling="cls")
        vectors = provider.embed(["a", "b"])
        expected = np.array([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.float32)
        np.testing.assert_array_equal(vectors, expected)


class TestDimensionMismatch:
    def test_raises_provider_error_when_actual_width_differs(self) -> None:
        provider = _provider(dimension=999)
        with pytest.raises(ProviderError, match="999"):
            provider.embed(["a"])

    def test_error_context_names_expected_and_actual(self) -> None:
        provider = _provider(dimension=999)
        with pytest.raises(ProviderError) as exc_info:
            provider.embed(["a"])
        assert exc_info.value.context["expected_dimension"] == 999
        assert exc_info.value.context["actual_dimension"] == 4


class TestGenerationFailure:
    def test_model_forward_failure_is_wrapped_as_provider_error(
        self, fake_model: Any
    ) -> None:
        fake_model.side_effect = RuntimeError("out of memory")
        provider = _provider()
        with pytest.raises(ProviderError):
            provider.embed(["a"])


class TestDimensionProperty:
    def test_returns_the_configured_value_without_loading_the_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []
        monkeypatch.setattr(
            "openbtk.embeddings.huggingface.require",
            lambda module, extra: calls.append(module),
        )
        provider = _provider(dimension=768)
        assert provider.dimension == 768
        assert calls == []


class TestModelIdentityAndProvenance:
    def test_model_identity_uses_model_and_revision_separately(self) -> None:
        provider = _provider(model="org/model-name", revision="deadbeef")
        identity = provider.model_identity()
        assert identity.name == "org/model-name"
        assert identity.revision == "deadbeef"
        assert identity.source == "huggingface"

    def test_provenance_records_model_pooling_and_device(self) -> None:
        provider = _provider(model="org/model-name", pooling="cls", device="cuda")
        provenance = provider.provenance()
        assert provenance.config == {
            "model": "org/model-name",
            "pooling": "cls",
            "device": "cuda",
        }


class TestModelLaziness:
    def test_constructing_the_provider_does_not_load_the_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []
        monkeypatch.setattr(
            "openbtk.embeddings.huggingface.require",
            lambda module, extra: calls.append(module),
        )
        _provider()
        assert calls == []

    def test_the_model_is_loaded_only_once_across_calls(self, fake_model: Any) -> None:
        provider = _provider()
        provider.embed(["one"])
        provider.embed(["two"])
        assert fake_model.call_count == 2
        fake_model.to.assert_called_once()


class TestDeclaredAttributes:
    def test_sends_data_offsite_is_false(self) -> None:
        assert HuggingFaceEmbeddingProvider.sends_data_offsite is False

    def test_registered_under_the_expected_key(self) -> None:
        assert (
            HuggingFaceEmbeddingProvider.registry_key == "embedding.general.huggingface"
        )
