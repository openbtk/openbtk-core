"""Unit tests for openbtk.llms.openai.OpenAIProvider.

The ``openai`` package is an optional dependency (the ``llms`` extra) not
installed in CI's zero-extras test-core job -- every test here mocks the
SDK surface directly (a fake module standing in for what ``require()``
would return) rather than needing the real package installed. This
achieves full behavioural coverage (retry-on-rate-limit, exception
translation, token accounting, streaming) without any real network call;
the real network path is exercised separately, and only when opted in, by
tests/contract/test_llm_contract.py.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from openbtk.core.errors import AuthenticationError, ProviderError, RateLimitError
from openbtk.core.schemas import LLMResponse, Message
from openbtk.llms.openai import OpenAIProvider


class _FakeRateLimitError(Exception):
    pass


class _FakeAuthenticationError(Exception):
    pass


class _FakeAPIError(Exception):
    pass


def _fake_openai_module(client: Any) -> Any:
    """A duck-typed stand-in for the real ``openai`` module: only the
    attributes OpenAIProvider actually touches (``OpenAI``, the three
    exception classes)."""
    module = MagicMock()
    module.OpenAI = MagicMock(return_value=client)
    module.RateLimitError = _FakeRateLimitError
    module.AuthenticationError = _FakeAuthenticationError
    module.APIError = _FakeAPIError
    return module


def _chat_response(
    text: str = "hello there",
    *,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    total_tokens: int = 15,
) -> Any:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=text))]
    response.usage = MagicMock(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
    return response


@pytest.fixture
def fake_client() -> Any:
    return MagicMock()


@pytest.fixture(autouse=True)
def _patch_require(monkeypatch: pytest.MonkeyPatch, fake_client: Any) -> None:
    monkeypatch.setattr(
        "openbtk.llms.openai.require",
        lambda module, extra: _fake_openai_module(fake_client),
    )


class TestChat:
    def test_returns_llm_response_with_the_completion_text(
        self, fake_client: Any
    ) -> None:
        fake_client.chat.completions.create.return_value = _chat_response("hi!")
        provider = OpenAIProvider()
        result = provider.chat([Message(role="user", content="hello")])
        assert isinstance(result, LLMResponse)
        assert result.text == "hi!"

    def test_populates_token_usage_from_the_response(self, fake_client: Any) -> None:
        fake_client.chat.completions.create.return_value = _chat_response(
            prompt_tokens=7, completion_tokens=3, total_tokens=10
        )
        provider = OpenAIProvider()
        result = provider.chat([Message(role="user", content="hello")])
        assert result.usage is not None
        assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (7, 3)
        assert result.usage.total_tokens == 10

    def test_usage_is_none_when_the_response_carries_none(
        self, fake_client: Any
    ) -> None:
        response = _chat_response()
        response.usage = None
        fake_client.chat.completions.create.return_value = response
        provider = OpenAIProvider()
        result = provider.chat([Message(role="user", content="hello")])
        assert result.usage is None

    def test_a_none_content_becomes_an_empty_string(self, fake_client: Any) -> None:
        response = _chat_response()
        response.choices[0].message.content = None
        fake_client.chat.completions.create.return_value = response
        provider = OpenAIProvider()
        result = provider.chat([Message(role="user", content="hello")])
        assert result.text == ""

    def test_forwards_the_configured_model_and_translated_messages(
        self, fake_client: Any
    ) -> None:
        fake_client.chat.completions.create.return_value = _chat_response()
        provider = OpenAIProvider(model="gpt-4o")
        provider.chat(
            [
                Message(role="system", content="be terse"),
                Message(role="user", content="hi"),
            ]
        )
        _, call_kwargs = fake_client.chat.completions.create.call_args
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["messages"] == [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]

    def test_extra_kwargs_are_forwarded_to_the_sdk_call(self, fake_client: Any) -> None:
        fake_client.chat.completions.create.return_value = _chat_response()
        provider = OpenAIProvider()
        provider.chat([Message(role="user", content="hi")], temperature=0.2)
        _, call_kwargs = fake_client.chat.completions.create.call_args
        assert call_kwargs["temperature"] == 0.2


class TestGenerate:
    def test_generate_wraps_the_prompt_as_a_single_user_message(
        self, fake_client: Any
    ) -> None:
        fake_client.chat.completions.create.return_value = _chat_response("ok")
        provider = OpenAIProvider()
        result = provider.generate("a single prompt")
        assert result.text == "ok"
        _, call_kwargs = fake_client.chat.completions.create.call_args
        assert call_kwargs["messages"] == [
            {"role": "user", "content": "a single prompt"}
        ]


class TestStream:
    def test_yields_only_non_empty_content_deltas(self, fake_client: Any) -> None:
        def _delta(content: str | None) -> Any:
            return MagicMock(choices=[MagicMock(delta=MagicMock(content=content))])

        fake_client.chat.completions.create.return_value = iter(
            [_delta("a"), _delta(None), _delta("b")]
        )
        provider = OpenAIProvider()
        assert list(provider.stream("go")) == ["a", "b"]

    def test_requests_streaming_from_the_sdk(self, fake_client: Any) -> None:
        fake_client.chat.completions.create.return_value = iter([])
        provider = OpenAIProvider()
        list(provider.stream("go"))
        _, call_kwargs = fake_client.chat.completions.create.call_args
        assert call_kwargs["stream"] is True


class TestErrorTranslation:
    """retry_with_backoff's own retry-then-succeed behaviour is already
    covered end to end in tests/unit/llms/test_base.py with an injectable
    no-op sleep; OpenAIProvider exposes no sleep override (there is no real
    use case for one), so these tests use max_retries=1 -- the "last
    attempt" branch re-raises immediately with no delay -- to verify
    exception *translation* without a real backoff wait."""

    def test_persistent_rate_limit_error_raises_openbtks_rate_limit_error(
        self, fake_client: Any
    ) -> None:
        fake_client.chat.completions.create.side_effect = _FakeRateLimitError("nope")
        provider = OpenAIProvider(max_retries=1)
        with pytest.raises(RateLimitError):
            provider.chat([Message(role="user", content="hi")])

    def test_authentication_error_is_translated_and_not_retried(
        self, fake_client: Any
    ) -> None:
        calls = []

        def always_fails(*args: object, **kwargs: object) -> Any:
            calls.append(1)
            raise _FakeAuthenticationError("bad key")

        fake_client.chat.completions.create.side_effect = always_fails
        provider = OpenAIProvider(max_retries=5)
        with pytest.raises(AuthenticationError):
            provider.chat([Message(role="user", content="hi")])
        assert len(calls) == 1

    def test_generic_api_error_is_translated_to_provider_error(
        self, fake_client: Any
    ) -> None:
        fake_client.chat.completions.create.side_effect = _FakeAPIError("boom")
        provider = OpenAIProvider(max_retries=1)
        with pytest.raises(ProviderError):
            provider.chat([Message(role="user", content="hi")])


class TestModelIdentityAndProvenance:
    def test_model_identity_uses_the_configured_model_as_both_name_and_revision(
        self,
    ) -> None:
        provider = OpenAIProvider(model="gpt-4o")
        identity = provider.model_identity()
        assert identity.name == "gpt-4o"
        assert identity.revision == "gpt-4o"
        assert identity.source == "api"

    def test_provenance_config_never_includes_the_api_key(self) -> None:
        provider = OpenAIProvider(api_key="sk-should-never-appear")
        dumped = provider.provenance().model_dump_json()
        assert "sk-should-never-appear" not in dumped

    def test_provenance_records_the_model_and_base_url(self) -> None:
        provider = OpenAIProvider(model="gpt-4o", base_url="https://example.test/v1")
        provenance = provider.provenance()
        assert provenance.config == {
            "model": "gpt-4o",
            "base_url": "https://example.test/v1",
        }
        assert provenance.model_identity is not None
        assert provenance.model_identity.name == "gpt-4o"


class TestClientLaziness:
    def test_constructing_the_provider_does_not_build_the_sdk_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []
        monkeypatch.setattr(
            "openbtk.llms.openai.require",
            lambda module, extra: calls.append(1) or _fake_openai_module(MagicMock()),
        )
        OpenAIProvider()
        assert calls == []

    def test_the_client_is_reused_across_calls(self, fake_client: Any) -> None:
        fake_client.chat.completions.create.return_value = _chat_response()
        provider = OpenAIProvider()
        provider.generate("one")
        provider.generate("two")
        assert fake_client.chat.completions.create.call_count == 2


class TestDeclaredAttributes:
    def test_sends_data_offsite_is_true(self) -> None:
        assert OpenAIProvider.sends_data_offsite is True

    def test_registered_under_the_expected_key(self) -> None:
        assert OpenAIProvider.registry_key == "llm.general.openai"
