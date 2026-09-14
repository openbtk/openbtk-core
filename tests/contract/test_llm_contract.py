"""Shared contract every registered BaseLLMProvider must satisfy."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel

from openbtk.core.registry import LLM_REGISTRY
from openbtk.core.schemas import LLMResponse, Message

if TYPE_CHECKING:
    from openbtk.core.base import BaseLLMProvider


def _new_instance(key: str) -> BaseLLMProvider:
    return LLM_REGISTRY.create(key)


class _Answer(BaseModel):
    value: str


@pytest.mark.parametrize("key", LLM_REGISTRY.list_keys())
class TestLLMContract:
    def test_declares_sends_data_offsite(self, key: str) -> None:
        provider = _new_instance(key)
        assert isinstance(provider.sends_data_offsite, bool)

    def test_generate_returns_llm_response(self, key: str) -> None:
        provider = _new_instance(key)
        result = provider.generate("hello")
        assert isinstance(result, LLMResponse)
        assert isinstance(result.text, str)

    def test_stream_returns_iterator_of_strings(self, key: str) -> None:
        provider = _new_instance(key)
        result = provider.stream("hello")
        assert isinstance(result, Iterator)
        chunks = list(result)
        assert all(isinstance(c, str) for c in chunks)

    def test_chat_returns_llm_response(self, key: str) -> None:
        provider = _new_instance(key)
        result = provider.chat([Message(role="user", content="hi")])
        assert isinstance(result, LLMResponse)

    def test_model_identity_default_raises_unless_overridden(self, key: str) -> None:
        provider = _new_instance(key)
        try:
            identity = provider.model_identity()
        except NotImplementedError:
            return
        assert identity.name and identity.revision and identity.source

    def test_provenance_is_serialisable(self, key: str) -> None:
        provider = _new_instance(key)
        dumped = provider.provenance().model_dump_json()
        assert isinstance(dumped, str) and len(dumped) > 0


def test_structured_default_parses_json_response() -> None:
    """The default structured() implementation (chat() + JSON validation)
    is only meaningfully testable against a provider whose chat() actually
    returns valid JSON for a given prompt -- the reference LLM's echo
    behaviour doesn't produce JSON, so this is tested directly rather than
    parametrized across every future provider, whose chat() responses are
    entirely provider-specific and cannot be assumed to be JSON at all."""

    class _JsonEchoLLM:
        def chat(self, messages: list[Message], **kwargs: object) -> LLMResponse:
            return LLMResponse(text='{"value": "ok"}')

    from openbtk.core.base import BaseLLMProvider

    result = BaseLLMProvider.structured(
        _JsonEchoLLM(),  # type: ignore[arg-type]
        [Message(role="user", content="respond in json")],
        _Answer,
    )
    assert isinstance(result, _Answer)
    assert result.value == "ok"
