"""Unit tests for openbtk.llms.openai_compatible.OpenAICompatibleProvider.

Unlike the OpenAI/Anthropic providers, this one is built directly on
``httpx`` -- a core, always-installed dependency -- so these tests use a
real ``httpx.MockTransport`` instead of mocking an optional SDK module.
That exercises the actual request/response/error-mapping code path
end-to-end (request building, JSON parsing, status-code mapping) rather
than mocking it away.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from openbtk.core.errors import AuthenticationError, ProviderError, RateLimitError
from openbtk.core.schemas import LLMResponse, Message
from openbtk.llms.openai_compatible import OpenAICompatibleProvider


def _chat_completion_payload(
    text: str = "hello there",
    *,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    total_tokens: int = 15,
) -> dict[str, Any]:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


def _provider_with_transport(handler: Any, **kwargs: Any) -> OpenAICompatibleProvider:
    kwargs.setdefault("model", "local-model")
    provider = OpenAICompatibleProvider(base_url="http://localhost:8000/v1", **kwargs)
    # Force client construction, then swap its transport for the mock --
    # the only way to intercept a request without a real endpoint while
    # still exercising the provider's own lazy-client-construction path.
    client = provider._get_client()
    client._transport = httpx.MockTransport(handler)
    return provider


class TestChat:
    def test_returns_llm_response_with_the_completion_text(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_chat_completion_payload("hi!"))

        provider = _provider_with_transport(handler)
        result = provider.chat([Message(role="user", content="hello")])
        assert isinstance(result, LLMResponse)
        assert result.text == "hi!"

    def test_populates_token_usage_from_the_payload(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=_chat_completion_payload(
                    prompt_tokens=7, completion_tokens=3, total_tokens=10
                ),
            )

        provider = _provider_with_transport(handler)
        result = provider.chat([Message(role="user", content="hi")])
        assert result.usage is not None
        assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (7, 3)

    def test_usage_is_none_when_the_payload_has_none(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = _chat_completion_payload()
            del payload["usage"]
            return httpx.Response(200, json=payload)

        provider = _provider_with_transport(handler)
        result = provider.chat([Message(role="user", content="hi")])
        assert result.usage is None

    def test_posts_the_model_and_translated_messages_to_chat_completions(
        self,
    ) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=_chat_completion_payload())

        provider = _provider_with_transport(handler, model="my-served-model")
        provider.chat([Message(role="user", content="hi")])
        assert captured[0].url.path == "/v1/chat/completions"
        body = json.loads(captured[0].content)
        assert body["model"] == "my-served-model"
        assert body["messages"] == [{"role": "user", "content": "hi"}]

    def test_sends_a_bearer_authorization_header_when_an_api_key_is_set(
        self,
    ) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=_chat_completion_payload())

        provider = _provider_with_transport(handler, api_key="test-key-value")
        provider.chat([Message(role="user", content="hi")])
        assert captured[0].headers["authorization"] == "Bearer test-key-value"

    def test_sends_no_authorization_header_without_an_api_key(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=_chat_completion_payload())

        provider = _provider_with_transport(handler)
        provider.chat([Message(role="user", content="hi")])
        assert "authorization" not in captured[0].headers


class TestGenerate:
    def test_generate_wraps_the_prompt_as_a_single_user_message(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=_chat_completion_payload("ok"))

        provider = _provider_with_transport(handler)
        result = provider.generate("a single prompt")
        assert result.text == "ok"
        body = json.loads(captured[0].content)
        assert body["messages"] == [{"role": "user", "content": "a single prompt"}]


class TestStream:
    def test_yields_deltas_from_server_sent_events(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            lines = [
                'data: {"choices": [{"delta": {"content": "a"}}]}',
                "",  # blank keep-alive line: not "data: "-prefixed, must be skipped
                'data: {"choices": [{"delta": {}}]}',
                'data: {"choices": [{"delta": {"content": "b"}}]}',
                "data: [DONE]",
            ]
            body = "\n".join(lines).encode()
            return httpx.Response(200, content=body)

        provider = _provider_with_transport(handler)
        assert list(provider.stream("go")) == ["a", "b"]

    def test_stream_ending_without_a_done_marker_still_yields_everything(
        self,
    ) -> None:
        """Some servers close the connection without ever sending the
        conventional "data: [DONE]" sentinel -- the loop must still end
        cleanly via normal iterator exhaustion, not only via the break."""

        def handler(request: httpx.Request) -> httpx.Response:
            body = b'data: {"choices": [{"delta": {"content": "only"}}]}'
            return httpx.Response(200, content=body)

        provider = _provider_with_transport(handler)
        assert list(provider.stream("go")) == ["only"]


class TestErrorTranslation:
    def test_429_is_translated_to_openbtks_rate_limit_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"error": "slow down"})

        provider = _provider_with_transport(handler, max_retries=1)
        with pytest.raises(RateLimitError):
            provider.chat([Message(role="user", content="hi")])

    @pytest.mark.parametrize("status", [401, 403])
    def test_401_and_403_are_translated_to_authentication_error(
        self, status: int
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status, json={"error": "unauthorized"})

        provider = _provider_with_transport(handler, max_retries=1)
        with pytest.raises(AuthenticationError):
            provider.chat([Message(role="user", content="hi")])

    def test_other_error_statuses_are_translated_to_provider_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "boom"})

        provider = _provider_with_transport(handler, max_retries=1)
        with pytest.raises(ProviderError):
            provider.chat([Message(role="user", content="hi")])

    def test_a_connection_error_is_translated_to_provider_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        provider = _provider_with_transport(handler, max_retries=1)
        with pytest.raises(ProviderError):
            provider.chat([Message(role="user", content="hi")])


class TestConstruction:
    def test_requires_a_non_empty_base_url(self) -> None:
        with pytest.raises(ValueError, match="base_url"):
            OpenAICompatibleProvider(base_url="", model="x")

    def test_trailing_slash_on_base_url_is_stripped(self) -> None:
        provider = OpenAICompatibleProvider(
            base_url="http://localhost:8000/v1/", model="x"
        )
        assert provider.provenance().config["base_url"] == "http://localhost:8000/v1"

    def test_constructing_the_provider_does_not_build_the_http_client(self) -> None:
        provider = OpenAICompatibleProvider(
            base_url="http://localhost:8000/v1", model="x"
        )
        assert provider._client is None


class TestModelIdentityAndProvenance:
    def test_model_identity_uses_the_configured_model(self) -> None:
        provider = OpenAICompatibleProvider(
            base_url="http://localhost:8000/v1", model="my-model"
        )
        identity = provider.model_identity()
        assert identity.name == "my-model"
        assert identity.revision == "my-model"
        assert identity.source == "api"

    def test_provenance_config_never_includes_the_api_key(self) -> None:
        provider = OpenAICompatibleProvider(
            base_url="http://localhost:8000/v1",
            model="x",
            api_key="should-never-appear",  # pragma: allowlist secret
        )
        dumped = provider.provenance().model_dump_json()
        assert "should-never-appear" not in dumped


class TestDeclaredAttributes:
    def test_sends_data_offsite_is_conservatively_true(self) -> None:
        assert OpenAICompatibleProvider.sends_data_offsite is True

    def test_registered_under_the_expected_key(self) -> None:
        assert OpenAICompatibleProvider.registry_key == "llm.general.openai_compatible"
