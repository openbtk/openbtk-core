"""Anthropic LLM provider -- a thin adapter over the official ``anthropic`` SDK.

Wraps, does not reinvent (docs/09_CODING_STANDARDS.md section 7). Two real
shape differences from :class:`~openbtk.llms.openai.OpenAIProvider`, not
copy-paste with names swapped:

* The Messages API takes ``system`` as its own top-level parameter, not a
  ``role: "system"`` message in the list -- a leading system ``Message``
  is split out of ``messages`` before the call, not passed through.
* ``max_tokens`` is a *required* parameter of ``messages.create`` (no SDK
  default), unlike OpenAI's Chat Completions -- this class supplies its own
  default so ``generate()``/``chat()`` work without the caller naming one
  every time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from openbtk.core._lazy import require
from openbtk.core.base import BaseLLMProvider
from openbtk.core.errors import AuthenticationError, ProviderError, RateLimitError
from openbtk.core.logging import get_logger
from openbtk.core.provenance import ComponentProvenance, ModelIdentity
from openbtk.core.registry import LLM_REGISTRY
from openbtk.core.schemas import LLMResponse, Message, TokenUsage
from openbtk.llms.base import retry_with_backoff

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

log = get_logger(__name__)

_DEFAULT_MODEL = "claude-opus-5"
_DEFAULT_MAX_TOKENS = 4096


@LLM_REGISTRY.register("llm.general.anthropic")
class AnthropicProvider(BaseLLMProvider):
    """Generate text via Anthropic's Messages API.

    Args:
        model: The Claude model id, e.g. ``"claude-opus-5"``.
        max_tokens: The Messages API's required output-token ceiling.
            Applied whenever a call does not supply its own.
        api_key: Forwarded to the SDK client. ``None`` (the default) lets
            the SDK resolve ``ANTHROPIC_API_KEY`` (or an ``ant auth
            login`` profile) -- never read or logged by this class.
        base_url: Forwarded to the SDK client. ``None`` uses Anthropic's
            own API.
        max_retries: Forwarded to :func:`~openbtk.llms.base.retry_with_backoff`
            as ``max_attempts``. Independent of the SDK's own internal
            retry setting (left at the SDK's default).

    No I/O happens in ``__init__`` (docs/09_CODING_STANDARDS.md rule 11):
    the SDK client is constructed lazily, on first real call.
    """

    sends_data_offsite: ClassVar[bool] = True

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 5,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._api_key = api_key
        self._base_url = base_url
        self._max_retries = max_retries
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            anthropic = require("anthropic", extra="llms")
            self._client = anthropic.Anthropic(
                api_key=self._api_key, base_url=self._base_url
            )
        return self._client

    def _call_with_retry(self, func: Callable[[], Any]) -> Any:
        """Run ``func`` (an Anthropic SDK call) with retry-on-rate-limit,
        translating the SDK's own exceptions to OpenBTK's first -- see
        this module's own docstring for why translation happens here
        rather than in :func:`retry_with_backoff`."""
        anthropic = require("anthropic", extra="llms")

        def wrapped() -> Any:
            try:
                return func()
            except anthropic.RateLimitError as e:
                raise RateLimitError(str(e)) from e
            except anthropic.AuthenticationError as e:
                raise AuthenticationError(str(e)) from e
            except anthropic.APIStatusError as e:
                raise ProviderError(str(e)) from e
            except anthropic.APIConnectionError as e:
                raise ProviderError(str(e)) from e

        return retry_with_backoff(wrapped, max_attempts=self._max_retries)

    @staticmethod
    def _split_system_message(
        messages: list[Message],
    ) -> tuple[str | None, list[Message]]:
        """The Messages API's ``system`` is a top-level parameter, not a
        message with ``role: "system"`` -- OpenBTK's ``Message`` still
        allows that role (it is shared across every provider), so a
        leading one is pulled out here rather than rejected."""
        if messages and messages[0].role == "system":
            return messages[0].content, messages[1:]
        return None, messages

    @staticmethod
    def _to_anthropic_messages(messages: list[Message]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    @staticmethod
    def _usage_from_response(response: Any) -> TokenUsage | None:
        usage = response.usage
        if usage is None:
            return None
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        return TokenUsage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

    @staticmethod
    def _text_from_response(response: Any) -> str:
        return "".join(block.text for block in response.content if block.type == "text")

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate a completion for a single prompt.

        Implemented via :meth:`chat` with a single user message.
        """
        return self.chat([Message(role="user", content=prompt)], **kwargs)

    def chat(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        system, rest = self._split_system_message(messages)
        client = self._get_client()
        kwargs.setdefault("max_tokens", self._max_tokens)
        if system is not None:
            kwargs.setdefault("system", system)
        response = self._call_with_retry(
            lambda: client.messages.create(
                model=self._model,
                messages=self._to_anthropic_messages(rest),
                **kwargs,
            )
        )
        return LLMResponse(
            text=self._text_from_response(response),
            usage=self._usage_from_response(response),
        )

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream a completion, token-by-token.

        Only opening the stream (``__enter__``, where the SDK's
        ``httpx``-based client actually issues the request) goes through
        :meth:`_call_with_retry` -- once tokens have started arriving,
        retrying would mean silently re-running the prompt and duplicating
        already-yielded output, which is worse than surfacing the error.
        The context manager is still always closed via ``finally``, retry
        or not, so a failure mid-stream never leaks the connection.
        """
        client = self._get_client()
        kwargs.setdefault("max_tokens", self._max_tokens)
        stream_context = client.messages.stream(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        stream = self._call_with_retry(stream_context.__enter__)
        try:
            yield from stream.text_stream
        finally:
            stream_context.__exit__(None, None, None)

    def model_identity(self) -> ModelIdentity:
        return ModelIdentity(name=self._model, revision=self._model, source="api")

    def provenance(self) -> ComponentProvenance:
        return (
            super()
            .provenance()
            .model_copy(
                update={
                    "config": {
                        "model": self._model,
                        "max_tokens": self._max_tokens,
                        "base_url": self._base_url,
                    },
                    "model_identity": self.model_identity(),
                }
            )
        )
