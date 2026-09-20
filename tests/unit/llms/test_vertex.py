"""``VertexAIProvider`` (FR-V-01) against a fake ``google.genai``: no Google call.
 ``tests/unit/llms/test_cloud_sdk_shapes.py`` checks the shapes against the real
SDK when it is installed."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from openbtk.core.errors import AuthenticationError, ProviderError, RateLimitError
from openbtk.core.retry import retry_with_backoff
from openbtk.core.schemas import LLMResponse, Message
from openbtk.llms import vertex
from openbtk.llms.vertex import VertexAIProvider

if TYPE_CHECKING:
    from collections.abc import Iterator


class _APIError(Exception):
    def __init__(self, code: int, status: str = "STATUS", message: str = "msg") -> None:
        super().__init__(f"{code} {status}. {message}")
        self.code = code
        self.status = status
        self.message = message


class _Record:
    """A stand-in for an SDK model: keeps the keyword arguments it was built with."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _Content(_Record):
    pass


class _Part(_Record):
    pass


class _Config(_Record):
    pass


class _FakeModels:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response: Any = SimpleNamespace(
            text="hi",
            usage_metadata=SimpleNamespace(
                prompt_token_count=3, candidates_token_count=2, total_token_count=5
            ),
        )
        self.chunks: list[Any] = []
        self.error: Exception | None = None
        self.errors_then_ok: list[Exception] = []

    def generate_content(self, **request: Any) -> Any:
        self.calls.append(request)
        if self.errors_then_ok:
            raise self.errors_then_ok.pop(0)
        if self.error:
            raise self.error
        return self.response

    def generate_content_stream(self, **request: Any) -> Any:
        self.calls.append(request)
        if self.error:
            raise self.error
        return iter(self.chunks)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    models = _FakeModels()
    made: list[dict[str, Any]] = []

    class _Genai:
        @staticmethod
        def Client(**kwargs: Any) -> Any:  # noqa: N802 - mirrors the SDK
            made.append(kwargs)
            return SimpleNamespace(models=models)

    modules = {
        "google.genai": _Genai,
        "google.genai.errors": SimpleNamespace(APIError=_APIError),
        "google.genai.types": SimpleNamespace(
            Content=_Content, Part=_Part, GenerateContentConfig=_Config
        ),
    }
    monkeypatch.setattr(vertex, "require", lambda name, extra: modules[name])
    monkeypatch.setattr(
        vertex,
        "retry_with_backoff",
        lambda func, max_attempts: retry_with_backoff(
            func, max_attempts=max_attempts, sleep=lambda s: None
        ),
    )
    yield {"models": models, "made": made}


def _provider(**kwargs: Any) -> VertexAIProvider:
    return VertexAIProvider(model="model-001", **kwargs)


class TestConstruction:
    def test_it_sends_data_offsite(self) -> None:
        assert VertexAIProvider.sends_data_offsite is True

    def test_constructing_does_no_io(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def explode(name: str, extra: str) -> None:
            raise AssertionError("constructing must not import or connect")

        monkeypatch.setattr(vertex, "require", explode)
        _provider()

    def test_the_client_is_vertex_and_built_once_lazily(
        self, fake: dict[str, Any]
    ) -> None:
        provider = _provider(project="p", location="europe-west4")
        assert fake["made"] == []
        provider.generate("a")
        provider.generate("b")
        assert fake["made"] == [
            {"vertexai": True, "project": "p", "location": "europe-west4"}
        ]

    def test_identity_and_provenance(self) -> None:
        provider = _provider(project="p", location="l")
        identity = provider.model_identity()
        assert (identity.name, identity.source) == ("model-001", "vertex")
        assert provider.provenance().config == {
            "model": "model-001",
            "project": "p",
            "location": "l",
        }


class TestRequests:
    def test_a_prompt_becomes_one_user_content(self, fake: dict[str, Any]) -> None:
        _provider().generate("hello")
        call = fake["models"].calls[0]
        assert call["model"] == "model-001" and call["config"] is None
        (content,) = call["contents"]
        assert content.role == "user" and content.parts[0].text == "hello"

    def test_system_becomes_the_instruction_and_assistant_becomes_model(
        self, fake: dict[str, Any]
    ) -> None:
        _provider().chat(
            [
                Message(role="system", content="Be brief."),
                Message(role="system", content="Cite sources."),
                Message(role="user", content="Q"),
                Message(role="assistant", content="A"),
            ]
        )
        call = fake["models"].calls[0]
        assert [c.role for c in call["contents"]] == ["user", "model"]
        assert call["config"].system_instruction == "Be brief.\n\nCite sources."

    def test_generation_settings_use_the_sdk_names(self, fake: dict[str, Any]) -> None:
        _provider().generate("x", max_tokens=10, temperature=0.3, top_p=0.8, stop=["E"])
        config = fake["models"].calls[0]["config"]
        assert (config.max_output_tokens, config.temperature) == (10, 0.3)
        assert (config.top_p, config.stop_sequences) == (0.8, ["E"])

    def test_none_settings_are_dropped(self, fake: dict[str, Any]) -> None:
        _provider().generate("x", temperature=None)
        assert fake["models"].calls[0]["config"] is None

    def test_an_unsupported_argument_is_refused_before_anything_is_sent(
        self, fake: dict[str, Any]
    ) -> None:
        with pytest.raises(ProviderError) as excinfo:
            _provider().generate("SECRET-PROMPT", top_k=3)
        assert "top_k" in str(excinfo.value)
        assert "SECRET-PROMPT" not in str(excinfo.value)
        assert fake["models"].calls == []


class TestResponses:
    def test_text_and_usage(self, fake: dict[str, Any]) -> None:
        result = _provider().generate("x")
        assert isinstance(result, LLMResponse) and result.text == "hi"
        assert result.usage is not None
        assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (3, 2)
        assert result.usage.total_tokens == 5

    def test_no_text_is_an_empty_string_and_no_usage_is_none(
        self, fake: dict[str, Any]
    ) -> None:
        fake["models"].response = SimpleNamespace(text=None, usage_metadata=None)
        result = _provider().generate("x")
        assert result.text == "" and result.usage is None


class TestStreaming:
    def test_chunks_with_text_are_yielded(self, fake: dict[str, Any]) -> None:
        fake["models"].chunks = [
            SimpleNamespace(text="Hel"),
            SimpleNamespace(text=None),
            SimpleNamespace(text=""),
            SimpleNamespace(text="lo"),
        ]
        assert list(_provider().stream("x")) == ["Hel", "lo"]

    def test_an_error_part_way_through_is_translated(
        self, fake: dict[str, Any]
    ) -> None:
        def broken() -> Iterator[Any]:
            yield SimpleNamespace(text="part")
            raise _APIError(429)

        fake["models"].generate_content_stream = lambda **_: broken()
        stream = _provider().stream("x")
        assert next(stream) == "part"
        with pytest.raises(RateLimitError):
            next(stream)


class TestErrors:
    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (429, RateLimitError),
            (401, AuthenticationError),
            (403, AuthenticationError),
            (400, ProviderError),
            (500, ProviderError),
        ],
    )
    def test_api_errors_are_translated_by_status_code(
        self, fake: dict[str, Any], code: int, expected: type[Exception]
    ) -> None:
        fake["models"].error = _APIError(code)
        with pytest.raises(expected) as excinfo:
            _provider(max_retries=1).generate("x")
        assert str(code) in str(excinfo.value)

    def test_unrelated_exceptions_are_not_swallowed(self, fake: dict[str, Any]) -> None:
        fake["models"].error = KeyError("bug")
        with pytest.raises(KeyError):
            _provider(max_retries=1).generate("x")

    def test_rate_limits_are_retried_then_succeed(self, fake: dict[str, Any]) -> None:
        fake["models"].errors_then_ok = [_APIError(429), _APIError(429)]
        assert _provider(max_retries=5).generate("x").text == "hi"
        assert len(fake["models"].calls) == 3

    def test_authentication_errors_are_not_retried(self, fake: dict[str, Any]) -> None:
        fake["models"].error = _APIError(403)
        with pytest.raises(AuthenticationError):
            _provider(max_retries=5).generate("x")
        assert len(fake["models"].calls) == 1

    def test_error_text_never_contains_the_prompt(self, fake: dict[str, Any]) -> None:
        fake["models"].error = _APIError(400, message="invalid request")
        with pytest.raises(ProviderError) as excinfo:
            _provider(max_retries=1).generate("SECRET-PROMPT-TEXT")
        assert "SECRET-PROMPT-TEXT" not in str(excinfo.value)
