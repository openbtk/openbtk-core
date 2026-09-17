"""Unit tests for openbtk.llms.huggingface.HuggingFaceLocalProvider.

Neither ``torch`` nor ``transformers`` needs to be installed to run these:
every test mocks both modules directly (fake tensor/streamer objects with
just enough real behaviour -- slicing, ``.shape``, iteration -- to exercise
the provider's actual logic), matching the same rationale as
test_openai.py and test_anthropic.py.
"""

from __future__ import annotations

import contextlib
import queue
from typing import Any
from unittest.mock import MagicMock

import pytest

from openbtk.core.errors import ProviderError
from openbtk.core.schemas import LLMResponse, Message
from openbtk.llms.huggingface import HuggingFaceLocalProvider


class _FakeTokenIds:
    """Just enough tensor-like behaviour for the provider's own slicing
    and length logic: a real Python list underneath, wrapped so slicing
    returns another _FakeTokenIds (a real list's slice does not)."""

    def __init__(self, data: list[int]) -> None:
        self.data = data

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, slice):
            return _FakeTokenIds(self.data[item])
        return self.data[item]

    def __len__(self) -> int:
        return len(self.data)

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.data),)


class _FakeBatchEncoding(dict[str, Any]):
    def to(self, device: str) -> _FakeBatchEncoding:
        return self


class _FakeStreamer:
    """A minimal stand-in for transformers.TextIteratorStreamer: a
    real queue-backed iterator, fed by the fake model.generate() call
    below, so stream()'s background-thread handoff is exercised for
    real rather than mocked away."""

    _SENTINEL = object()

    def __init__(self) -> None:
        self._queue: queue.Queue[Any] = queue.Queue()

    def put(self, token: str) -> None:
        self._queue.put(token)

    def end(self) -> None:
        self._queue.put(self._SENTINEL)

    def __iter__(self) -> _FakeStreamer:
        return self

    def __next__(self) -> str:
        item = self._queue.get()
        if item is self._SENTINEL:
            raise StopIteration
        return item


def _fake_torch_module() -> Any:
    module = MagicMock()
    module.no_grad = contextlib.nullcontext
    return module


def _fake_transformers_module(tokenizer: Any, model: Any) -> Any:
    module = MagicMock()
    module.AutoTokenizer.from_pretrained = MagicMock(return_value=tokenizer)
    module.AutoModelForCausalLM.from_pretrained = MagicMock(return_value=model)
    module.TextIteratorStreamer = MagicMock(
        side_effect=lambda *a, **kw: _FakeStreamer()
    )
    return module


@pytest.fixture
def fake_tokenizer() -> Any:
    tokenizer = MagicMock()
    tokenizer.chat_template = None
    tokenizer.side_effect = lambda prompt, return_tensors: _FakeBatchEncoding(
        input_ids=_FakeTokenIds([1, 2, 3])  # a 3-token "prompt"
    )
    tokenizer.decode = MagicMock(return_value="generated text")
    return tokenizer


@pytest.fixture
def fake_model(fake_tokenizer: Any) -> Any:
    model = MagicMock()
    model.to = MagicMock(return_value=model)

    def _generate(**kwargs: Any) -> list[_FakeTokenIds]:
        # prompt (3 tokens) + 2 "generated" tokens, batch size 1
        return [_FakeTokenIds([1, 2, 3, 4, 5])]

    model.generate = MagicMock(side_effect=_generate)
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

    monkeypatch.setattr("openbtk.llms.huggingface.require", fake_require)


def _provider(**kwargs: Any) -> HuggingFaceLocalProvider:
    kwargs.setdefault("model", "tiny/test-model")
    kwargs.setdefault("revision", "abc123def456")  # pragma: allowlist secret
    return HuggingFaceLocalProvider(**kwargs)


class TestGenerate:
    def test_returns_llm_response_with_the_decoded_text(self) -> None:
        provider = _provider()
        result = provider.generate("hello")
        assert isinstance(result, LLMResponse)
        assert result.text == "generated text"

    def test_token_usage_reflects_prompt_and_generated_lengths(self) -> None:
        provider = _provider()
        result = provider.generate("hello")
        assert result.usage is not None
        assert result.usage.prompt_tokens == 3
        assert result.usage.completion_tokens == 2
        assert result.usage.total_tokens == 5

    def test_only_the_generated_suffix_is_decoded_not_the_prompt(
        self, fake_tokenizer: Any
    ) -> None:
        provider = _provider()
        provider.generate("hello")
        decoded_ids = fake_tokenizer.decode.call_args[0][0]
        assert decoded_ids.data == [4, 5]

    def test_generation_failure_is_wrapped_as_a_provider_error(
        self, fake_model: Any
    ) -> None:
        fake_model.generate.side_effect = RuntimeError("out of memory")
        provider = _provider()
        with pytest.raises(ProviderError):
            provider.generate("hello")

    def test_caller_supplied_max_new_tokens_is_forwarded(self, fake_model: Any) -> None:
        provider = _provider(max_new_tokens=5)
        provider.generate("hello", max_new_tokens=99)
        assert fake_model.generate.call_args.kwargs["max_new_tokens"] == 99


class TestChat:
    def test_falls_back_to_a_role_prefixed_transcript_without_a_chat_template(
        self, fake_tokenizer: Any
    ) -> None:
        fake_tokenizer.chat_template = None
        provider = _provider()
        provider.chat(
            [
                Message(role="system", content="be terse"),
                Message(role="user", content="hi"),
            ]
        )
        call_prompt = fake_tokenizer.call_args[0][0]
        assert "system: be terse" in call_prompt
        assert "user: hi" in call_prompt
        assert call_prompt.endswith("assistant:")

    def test_uses_the_tokenizers_chat_template_when_present(
        self, fake_tokenizer: Any
    ) -> None:
        fake_tokenizer.chat_template = "a real jinja template"
        fake_tokenizer.apply_chat_template = MagicMock(return_value="templated prompt")
        provider = _provider()
        provider.chat([Message(role="user", content="hi")])
        call_prompt = fake_tokenizer.call_args[0][0]
        assert call_prompt == "templated prompt"
        fake_tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "hi"}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def test_chat_returns_an_llm_response(self) -> None:
        provider = _provider()
        result = provider.chat([Message(role="user", content="hi")])
        assert isinstance(result, LLMResponse)
        assert result.text == "generated text"


class TestStream:
    def test_yields_tokens_pushed_onto_the_streamer(self, fake_model: Any) -> None:
        def _generate_and_stream(**kwargs: Any) -> None:
            streamer = kwargs["streamer"]
            streamer.put("a")
            streamer.put("b")
            streamer.put("c")

        fake_model.generate.side_effect = _generate_and_stream
        provider = _provider()
        assert list(provider.stream("go")) == ["a", "b", "c"]

    def test_streamer_is_ended_even_when_generation_raises(
        self, fake_model: Any
    ) -> None:
        def _fail(**kwargs: Any) -> None:
            kwargs["streamer"].put("partial")
            raise RuntimeError("boom mid-generation")

        fake_model.generate.side_effect = _fail
        provider = _provider()
        with pytest.raises(ProviderError):
            list(provider.stream("go"))

    def test_stream_raises_immediately_if_generation_fails_before_any_token(
        self, fake_model: Any
    ) -> None:
        """The real bug this class's own docstring calls out: without
        streamer.end() in generate()'s finally, this would hang forever
        instead of raising."""

        def _fail_immediately(**kwargs: Any) -> None:
            raise RuntimeError("failed before producing anything")

        fake_model.generate.side_effect = _fail_immediately
        provider = _provider()
        with pytest.raises(ProviderError, match="failed before producing anything"):
            list(provider.stream("go"))


class TestModelIdentityAndProvenance:
    def test_model_identity_uses_model_and_revision_separately(self) -> None:
        provider = _provider(model="org/model-name", revision="deadbeef")
        identity = provider.model_identity()
        assert identity.name == "org/model-name"
        assert identity.revision == "deadbeef"
        assert identity.source == "huggingface"

    def test_provenance_records_model_and_device(self) -> None:
        provider = _provider(model="org/model-name", device="cuda")
        provenance = provider.provenance()
        assert provenance.config == {"model": "org/model-name", "device": "cuda"}


class TestModelLaziness:
    def test_constructing_the_provider_does_not_load_the_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []
        monkeypatch.setattr(
            "openbtk.llms.huggingface.require",
            lambda module, extra: calls.append(module),
        )
        _provider()
        assert calls == []

    def test_the_model_is_loaded_only_once_across_calls(self, fake_model: Any) -> None:
        provider = _provider()
        provider.generate("one")
        provider.generate("two")
        assert fake_model.generate.call_count == 2
        # to() is called once, at load time, not once per generate() call
        fake_model.to.assert_called_once()


class TestDeclaredAttributes:
    def test_sends_data_offsite_is_false(self) -> None:
        assert HuggingFaceLocalProvider.sends_data_offsite is False

    def test_registered_under_the_expected_key(self) -> None:
        assert HuggingFaceLocalProvider.registry_key == "llm.general.huggingface_local"
