"""Unit tests for openbtk.llms.anthropic.AnthropicProvider.

Same rationale as tests/unit/llms/test_openai.py: the ``anthropic`` package
is optional (the ``llms`` extra), so every test mocks the SDK surface
directly instead of needing it installed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from openbtk.core.errors import AuthenticationError, ProviderError, RateLimitError
from openbtk.core.schemas import LLMResponse, Message
from openbtk.llms.anthropic import AnthropicProvider


class _FakeRateLimitError(Exception):
    pass


class _FakeAuthenticationError(Exception):
    pass


class _FakeAPIStatusError(Exception):
    pass


class _FakeAPIConnectionError(Exception):
    pass


def _fake_anthropic_module(client: Any) -> Any:
    module = MagicMock()
    module.Anthropic = MagicMock(return_value=client)
    module.RateLimitError = _FakeRateLimitError
    module.AuthenticationError = _FakeAuthenticationError
    module.APIStatusError = _FakeAPIStatusError
    module.APIConnectionError = _FakeAPIConnectionError
    return module


def _text_block(text: str) -> Any:
    return MagicMock(type="text", text=text)


def _message_response(
    text: str = "hello there", *, input_tokens: int = 10, output_tokens: int = 5
) -> Any:
    response = MagicMock()
    response.content = [_text_block(text)]
    response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return response


@pytest.fixture
def fake_client() -> Any:
    return MagicMock()


@pytest.fixture(autouse=True)
def _patch_require(monkeypatch: pytest.MonkeyPatch, fake_client: Any) -> None:
    monkeypatch.setattr(
        "openbtk.llms.anthropic.require",
        lambda module, extra: _fake_anthropic_module(fake_client),
    )


class TestChat:
    def test_returns_llm_response_with_the_completion_text(
        self, fake_client: Any
    ) -> None:
        fake_client.messages.create.return_value = _message_response("hi!")
        provider = AnthropicProvider()
        result = provider.chat([Message(role="user", content="hello")])
        assert isinstance(result, LLMResponse)
        assert result.text == "hi!"

    def test_concatenates_multiple_text_blocks(self, fake_client: Any) -> None:
        response = _message_response()
        response.content = [_text_block("foo"), _text_block("bar")]
        fake_client.messages.create.return_value = response
        provider = AnthropicProvider()
        result = provider.chat([Message(role="user", content="hi")])
        assert result.text == "foobar"

    def test_populates_token_usage_as_prompt_plus_completion_equals_total(
        self, fake_client: Any
    ) -> None:
        fake_client.messages.create.return_value = _message_response(
            input_tokens=7, output_tokens=3
        )
        provider = AnthropicProvider()
        result = provider.chat([Message(role="user", content="hi")])
        assert result.usage is not None
        assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (7, 3)
        assert result.usage.total_tokens == 10

    def test_a_leading_system_message_is_split_into_the_top_level_param(
        self, fake_client: Any
    ) -> None:
        fake_client.messages.create.return_value = _message_response()
        provider = AnthropicProvider()
        provider.chat(
            [
                Message(role="system", content="be terse"),
                Message(role="user", content="hi"),
            ]
        )
        _, call_kwargs = fake_client.messages.create.call_args
        assert call_kwargs["system"] == "be terse"
        assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]

    def test_no_system_key_is_sent_without_a_leading_system_message(
        self, fake_client: Any
    ) -> None:
        fake_client.messages.create.return_value = _message_response()
        provider = AnthropicProvider()
        provider.chat([Message(role="user", content="hi")])
        _, call_kwargs = fake_client.messages.create.call_args
        assert "system" not in call_kwargs

    def test_max_tokens_defaults_when_the_caller_does_not_supply_one(
        self, fake_client: Any
    ) -> None:
        fake_client.messages.create.return_value = _message_response()
        provider = AnthropicProvider(max_tokens=777)
        provider.chat([Message(role="user", content="hi")])
        _, call_kwargs = fake_client.messages.create.call_args
        assert call_kwargs["max_tokens"] == 777

    def test_caller_supplied_max_tokens_overrides_the_default(
        self, fake_client: Any
    ) -> None:
        fake_client.messages.create.return_value = _message_response()
        provider = AnthropicProvider(max_tokens=777)
        provider.chat([Message(role="user", content="hi")], max_tokens=42)
        _, call_kwargs = fake_client.messages.create.call_args
        assert call_kwargs["max_tokens"] == 42

    def test_usage_is_none_when_the_response_carries_none(
        self, fake_client: Any
    ) -> None:
        response = _message_response()
        response.usage = None
        fake_client.messages.create.return_value = response
        provider = AnthropicProvider()
        result = provider.chat([Message(role="user", content="hi")])
        assert result.usage is None

    def test_forwards_the_configured_model(self, fake_client: Any) -> None:
        fake_client.messages.create.return_value = _message_response()
        provider = AnthropicProvider(model="claude-opus-5")
        provider.chat([Message(role="user", content="hi")])
        _, call_kwargs = fake_client.messages.create.call_args
        assert call_kwargs["model"] == "claude-opus-5"


class TestGenerate:
    def test_generate_wraps_the_prompt_as_a_single_user_message(
        self, fake_client: Any
    ) -> None:
        fake_client.messages.create.return_value = _message_response("ok")
        provider = AnthropicProvider()
        result = provider.generate("a single prompt")
        assert result.text == "ok"
        _, call_kwargs = fake_client.messages.create.call_args
        assert call_kwargs["messages"] == [
            {"role": "user", "content": "a single prompt"}
        ]


class TestStream:
    def test_yields_the_text_stream_contents(self, fake_client: Any) -> None:
        stream_context = MagicMock()
        stream_obj = MagicMock()
        stream_obj.text_stream = iter(["a", "b", "c"])
        stream_context.__enter__ = MagicMock(return_value=stream_obj)
        stream_context.__exit__ = MagicMock(return_value=False)
        fake_client.messages.stream.return_value = stream_context

        provider = AnthropicProvider()
        assert list(provider.stream("go")) == ["a", "b", "c"]
        stream_context.__exit__.assert_called_once()

    def test_exit_is_called_even_if_iteration_raises(self, fake_client: Any) -> None:
        stream_context = MagicMock()
        stream_obj = MagicMock()

        def _exploding() -> Any:
            yield "a"
            raise RuntimeError("boom")

        stream_obj.text_stream = _exploding()
        stream_context.__enter__ = MagicMock(return_value=stream_obj)
        stream_context.__exit__ = MagicMock(return_value=False)
        fake_client.messages.stream.return_value = stream_context

        provider = AnthropicProvider()
        with pytest.raises(RuntimeError):
            list(provider.stream("go"))
        stream_context.__exit__.assert_called_once()

    def test_rate_limit_error_opening_the_stream_is_translated(
        self, fake_client: Any
    ) -> None:
        stream_context = MagicMock()
        stream_context.__enter__ = MagicMock(side_effect=_FakeRateLimitError("nope"))
        fake_client.messages.stream.return_value = stream_context

        provider = AnthropicProvider(max_retries=1)
        with pytest.raises(RateLimitError):
            list(provider.stream("go"))


class TestErrorTranslation:
    """See test_openai.py's TestErrorTranslation docstring: max_retries=1
    avoids a real backoff wait while still exercising translation."""

    def test_persistent_rate_limit_error_raises_openbtks_rate_limit_error(
        self, fake_client: Any
    ) -> None:
        fake_client.messages.create.side_effect = _FakeRateLimitError("nope")
        provider = AnthropicProvider(max_retries=1)
        with pytest.raises(RateLimitError):
            provider.chat([Message(role="user", content="hi")])

    def test_authentication_error_is_translated_and_not_retried(
        self, fake_client: Any
    ) -> None:
        calls = []

        def always_fails(*args: object, **kwargs: object) -> Any:
            calls.append(1)
            raise _FakeAuthenticationError("bad key")

        fake_client.messages.create.side_effect = always_fails
        provider = AnthropicProvider(max_retries=5)
        with pytest.raises(AuthenticationError):
            provider.chat([Message(role="user", content="hi")])
        assert len(calls) == 1

    def test_api_status_error_is_translated_to_provider_error(
        self, fake_client: Any
    ) -> None:
        fake_client.messages.create.side_effect = _FakeAPIStatusError("boom")
        provider = AnthropicProvider(max_retries=1)
        with pytest.raises(ProviderError):
            provider.chat([Message(role="user", content="hi")])

    def test_api_connection_error_is_translated_to_provider_error(
        self, fake_client: Any
    ) -> None:
        fake_client.messages.create.side_effect = _FakeAPIConnectionError("boom")
        provider = AnthropicProvider(max_retries=1)
        with pytest.raises(ProviderError):
            provider.chat([Message(role="user", content="hi")])


class TestModelIdentityAndProvenance:
    def test_model_identity_uses_the_configured_model(self) -> None:
        provider = AnthropicProvider(model="claude-opus-5")
        identity = provider.model_identity()
        assert identity.name == "claude-opus-5"
        assert identity.revision == "claude-opus-5"
        assert identity.source == "api"

    def test_provenance_config_never_includes_the_api_key(self) -> None:
        provider = AnthropicProvider(api_key="should-never-appear")
        dumped = provider.provenance().model_dump_json()
        assert "should-never-appear" not in dumped

    def test_provenance_records_model_and_max_tokens(self) -> None:
        provider = AnthropicProvider(model="claude-opus-5", max_tokens=999)
        provenance = provider.provenance()
        assert provenance.config["model"] == "claude-opus-5"
        assert provenance.config["max_tokens"] == 999


class TestClientLaziness:
    def test_constructing_the_provider_does_not_build_the_sdk_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []
        monkeypatch.setattr(
            "openbtk.llms.anthropic.require",
            lambda module, extra: (
                calls.append(1) or _fake_anthropic_module(MagicMock())
            ),
        )
        AnthropicProvider()
        assert calls == []

    def test_the_client_is_reused_across_calls(self, fake_client: Any) -> None:
        fake_client.messages.create.return_value = _message_response()
        provider = AnthropicProvider()
        provider.generate("one")
        provider.generate("two")
        assert fake_client.messages.create.call_count == 2


class TestDeclaredAttributes:
    def test_sends_data_offsite_is_true(self) -> None:
        assert AnthropicProvider.sends_data_offsite is True

    def test_registered_under_the_expected_key(self) -> None:
        assert AnthropicProvider.registry_key == "llm.general.anthropic"
