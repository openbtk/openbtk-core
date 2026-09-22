"""``BedrockProvider`` (FR-V-01) against a fake ``boto3``/``botocore``: no AWS call. The
request and response shapes were read from botocore's own service definition;
``tests/unit/llms/test_cloud_sdk_shapes.py`` checks them against the real libraries when
those are installed."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from openbtk.core.errors import AuthenticationError, ProviderError, RateLimitError
from openbtk.core.provenance import ComponentProvenance
from openbtk.core.retry import retry_with_backoff
from openbtk.core.schemas import LLMResponse, Message
from openbtk.llms import bedrock
from openbtk.llms.bedrock import BedrockProvider

if TYPE_CHECKING:
    from collections.abc import Iterator


class _ClientError(Exception):
    def __init__(self, code: str, message: str = "msg") -> None:
        super().__init__(f"{code}: {message}")
        self.response = {"Error": {"Code": code, "Message": message}}


class _NoCredentialsError(Exception):
    pass


class _BotoCoreError(Exception):
    pass


class _FakeClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.response: Any = {
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5},
        }
        self.events: list[Any] = []
        self.error: Exception | None = None
        self.errors_then_ok: list[Exception] = []

    def converse(self, **request: Any) -> Any:
        self.requests.append(request)
        if self.errors_then_ok:
            raise self.errors_then_ok.pop(0)
        if self.error:
            raise self.error
        return self.response

    def converse_stream(self, **request: Any) -> Any:
        self.requests.append(request)
        if self.error:
            raise self.error
        return {"stream": iter(self.events)}


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    client = _FakeClient()
    made: list[dict[str, Any]] = []

    class _Boto3:
        @staticmethod
        def client(service: str, **kwargs: Any) -> _FakeClient:
            made.append({"service": service, **kwargs})
            return client

    class _Exceptions:
        ClientError = _ClientError
        NoCredentialsError = _NoCredentialsError
        BotoCoreError = _BotoCoreError

    modules = {"boto3": _Boto3, "botocore.exceptions": _Exceptions}
    monkeypatch.setattr(bedrock, "require", lambda name, extra: modules[name])
    monkeypatch.setattr(
        bedrock,
        "retry_with_backoff",
        lambda func, max_attempts: retry_with_backoff(
            func, max_attempts=max_attempts, sleep=lambda s: None
        ),
    )
    yield {"client": client, "made": made}


def _provider(**kwargs: Any) -> BedrockProvider:
    return BedrockProvider(model="vendor.model-v1:0", **kwargs)


class TestConstruction:
    def test_it_sends_data_offsite(self) -> None:
        assert BedrockProvider.sends_data_offsite is True

    def test_constructing_does_no_io(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def explode(name: str, extra: str) -> None:
            raise AssertionError("constructing must not import or connect")

        monkeypatch.setattr(bedrock, "require", explode)
        _provider()

    def test_the_client_is_built_once_lazily_with_the_region(
        self, fake: dict[str, Any]
    ) -> None:
        provider = _provider(region_name="eu-west-1")
        assert fake["made"] == []
        provider.generate("a")
        provider.generate("b")
        assert fake["made"] == [
            {"service": "bedrock-runtime", "region_name": "eu-west-1"}
        ]

    def test_identity_and_provenance(self) -> None:
        provider = _provider(region_name="us-east-1")
        identity = provider.model_identity()
        assert (identity.name, identity.source) == ("vendor.model-v1:0", "bedrock")
        provenance = provider.provenance()
        assert isinstance(provenance, ComponentProvenance)
        assert provenance.config == {
            "model": "vendor.model-v1:0",
            "region_name": "us-east-1",
        }


class TestRequests:
    def test_a_prompt_becomes_one_user_message(self, fake: dict[str, Any]) -> None:
        _provider().generate("hello")
        (request,) = fake["client"].requests
        assert request == {
            "modelId": "vendor.model-v1:0",
            "messages": [{"role": "user", "content": [{"text": "hello"}]}],
        }

    def test_system_messages_move_to_the_system_field(
        self, fake: dict[str, Any]
    ) -> None:
        _provider().chat(
            [
                Message(role="system", content="Be brief."),
                Message(role="user", content="Q"),
                Message(role="assistant", content="A"),
                Message(role="user", content="Q2"),
            ]
        )
        request = fake["client"].requests[0]
        assert request["system"] == [{"text": "Be brief."}]
        assert [m["role"] for m in request["messages"]] == ["user", "assistant", "user"]

    def test_generation_settings_map_to_inference_config(
        self, fake: dict[str, Any]
    ) -> None:
        _provider().generate(
            "x", max_tokens=50, temperature=0.2, top_p=0.9, stop=["END"]
        )
        assert fake["client"].requests[0]["inferenceConfig"] == {
            "maxTokens": 50,
            "temperature": 0.2,
            "topP": 0.9,
            "stopSequences": ["END"],
        }

    def test_none_settings_are_dropped(self, fake: dict[str, Any]) -> None:
        _provider().generate("x", temperature=None)
        assert "inferenceConfig" not in fake["client"].requests[0]

    def test_an_unsupported_argument_is_refused_naming_only_the_argument(
        self, fake: dict[str, Any]
    ) -> None:
        with pytest.raises(ProviderError) as excinfo:
            _provider().generate("SECRET-PROMPT", top_k=5, seed=1)
        assert "seed" in str(excinfo.value) and "top_k" in str(excinfo.value)
        assert "SECRET-PROMPT" not in str(excinfo.value)
        assert fake["client"].requests == []  # nothing was sent


class TestResponses:
    def test_text_and_usage(self, fake: dict[str, Any]) -> None:
        result = _provider().generate("x")
        assert isinstance(result, LLMResponse) and result.text == "hi"
        assert result.usage is not None
        assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (3, 2)
        assert result.usage.total_tokens == 5

    def test_several_text_blocks_are_joined_and_other_blocks_ignored(
        self, fake: dict[str, Any]
    ) -> None:
        fake["client"].response = {
            "output": {
                "message": {
                    "content": [
                        {"text": "a"},
                        {"toolUse": {"name": "t"}},
                        {"text": "b"},
                    ]
                }
            }
        }
        assert _provider().generate("x").text == "ab"

    def test_a_missing_usage_is_none_not_zero(self, fake: dict[str, Any]) -> None:
        fake["client"].response = {"output": {"message": {"content": [{"text": "t"}]}}}
        assert _provider().generate("x").usage is None

    def test_an_empty_output_is_an_empty_string(self, fake: dict[str, Any]) -> None:
        fake["client"].response = {"output": {}}
        assert _provider().generate("x").text == ""


class TestStreaming:
    def test_text_deltas_are_yielded(self, fake: dict[str, Any]) -> None:
        fake["client"].events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"text": "Hel"}, "contentBlockIndex": 0}},
            {"contentBlockDelta": {"delta": {"text": ""}}},
            {"contentBlockDelta": {"delta": {"text": "lo"}}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
        assert list(_provider().stream("x")) == ["Hel", "lo"]

    def test_a_streaming_request_uses_converse_stream_with_the_same_shape(
        self, fake: dict[str, Any]
    ) -> None:
        list(_provider().stream("hi", max_tokens=5))
        request = fake["client"].requests[0]
        assert request["inferenceConfig"] == {"maxTokens": 5}
        assert request["messages"][0]["content"] == [{"text": "hi"}]

    def test_an_error_arriving_mid_stream_is_translated(
        self, fake: dict[str, Any]
    ) -> None:
        def broken() -> Iterator[Any]:
            yield {"contentBlockDelta": {"delta": {"text": "part"}}}
            raise _ClientError("ThrottlingException", "slow down")

        fake["client"].converse_stream = lambda **_: {"stream": broken()}
        stream = _provider().stream("x")
        assert next(stream) == "part"
        with pytest.raises(RateLimitError):
            next(stream)


class TestErrors:
    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("ThrottlingException", RateLimitError),
            ("ServiceQuotaExceededException", RateLimitError),
            ("AccessDeniedException", AuthenticationError),
            ("ExpiredTokenException", AuthenticationError),
            ("UnrecognizedClientException", AuthenticationError),
            ("ValidationException", ProviderError),
            ("ModelNotReadyException", ProviderError),
        ],
    )
    def test_service_errors_are_translated_by_code(
        self, fake: dict[str, Any], code: str, expected: type[Exception]
    ) -> None:
        fake["client"].error = _ClientError(code)
        with pytest.raises(expected) as excinfo:
            _provider(max_retries=1).generate("x")
        assert code in str(excinfo.value)

    def test_missing_credentials_are_an_authentication_error(
        self, fake: dict[str, Any]
    ) -> None:
        fake["client"].error = _NoCredentialsError("Unable to locate credentials")
        with pytest.raises(AuthenticationError, match="No AWS credentials"):
            _provider(max_retries=1).generate("x")

    def test_other_botocore_errors_become_provider_errors(
        self, fake: dict[str, Any]
    ) -> None:
        fake["client"].error = _BotoCoreError("connection reset")
        with pytest.raises(ProviderError, match="_BotoCoreError"):
            _provider(max_retries=1).generate("x")

    def test_unrelated_exceptions_are_not_swallowed(self, fake: dict[str, Any]) -> None:
        fake["client"].error = KeyError("bug")
        with pytest.raises(KeyError):
            _provider(max_retries=1).generate("x")

    def test_throttling_is_retried_then_succeeds(self, fake: dict[str, Any]) -> None:
        fake["client"].errors_then_ok = [_ClientError("ThrottlingException")] * 2
        assert _provider(max_retries=5).generate("x").text == "hi"
        assert len(fake["client"].requests) == 3

    def test_authentication_errors_are_not_retried(self, fake: dict[str, Any]) -> None:
        fake["client"].error = _ClientError("AccessDeniedException")
        with pytest.raises(AuthenticationError):
            _provider(max_retries=5).generate("x")
        assert len(fake["client"].requests) == 1

    def test_error_text_never_contains_the_prompt(self, fake: dict[str, Any]) -> None:
        fake["client"].error = _ClientError("ValidationException", "bad request")
        with pytest.raises(ProviderError) as excinfo:
            _provider(max_retries=1).generate("SECRET-PROMPT-TEXT")
        assert "SECRET-PROMPT-TEXT" not in str(excinfo.value)
