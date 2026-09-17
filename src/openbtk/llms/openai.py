"""OpenAI LLM provider -- a thin adapter over the official ``openai`` SDK.

Wraps, does not reinvent (docs/09_CODING_STANDARDS.md section 7): all the
real work -- HTTP, auth, SSE streaming -- is the SDK's. This module's own
job is translating the SDK's request/response/exception shapes to
OpenBTK's (``Message``/``LLMResponse``/``TokenUsage``, and the
``RateLimitError``/``AuthenticationError``/``ProviderError`` hierarchy),
and applying :func:`openbtk.llms.base.retry_with_backoff` around every
call.
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

_DEFAULT_MODEL = "gpt-4o-mini"


@LLM_REGISTRY.register("llm.general.openai")
class OpenAIProvider(BaseLLMProvider):
    """Generate text via OpenAI's Chat Completions API.

    Args:
        model: The OpenAI model id, e.g. ``"gpt-4o-mini"`` or a dated
            snapshot such as ``"gpt-4o-mini-2024-07-18"`` for a genuinely
            pinned revision (docs/adr FR-P-05: never a floating tag --
            enforced by :class:`~openbtk.core.provenance.ModelIdentity`
            only for the literal known tag words, so an undated model
            name like the default here is accepted, but is the vendor's
            own moving target, not this project's).
        api_key: Forwarded to the SDK client. ``None`` (the default) lets
            the SDK resolve ``OPENAI_API_KEY`` from the environment --
            never read or logged by this class itself.
        base_url: Forwarded to the SDK client. ``None`` uses OpenAI's own
            API. Set this to point the *official* SDK at a self-hosted
            OpenAI-compatible endpoint instead of using
            :class:`~openbtk.llms.openai_compatible.OpenAICompatibleProvider`
            -- both are valid; the compatible provider exists for the
            common case of not wanting the ``openai`` package as a
            dependency at all.
        max_retries: Forwarded to :func:`~openbtk.llms.base.retry_with_backoff`
            as ``max_attempts`` for rate-limit retries. Independent of the
            SDK's own internal retry setting (left at the SDK's default).

    No I/O happens in ``__init__`` (docs/09_CODING_STANDARDS.md rule 11):
    the SDK client is constructed lazily, on first real call.
    """

    sends_data_offsite: ClassVar[bool] = True

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 5,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._max_retries = max_retries
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            openai = require("openai", extra="llms")
            self._client = openai.OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def _call_with_retry(self, func: Callable[[], Any]) -> Any:
        """Run ``func`` (an OpenAI SDK call) with retry-on-rate-limit,
        translating the SDK's own exceptions to OpenBTK's first -- see
        this module's own docstring for why translation happens here
        rather than in :func:`retry_with_backoff` (which only knows
        about OpenBTK's exception hierarchy, not any one SDK's)."""
        openai = require("openai", extra="llms")

        def wrapped() -> Any:
            try:
                return func()
            except openai.RateLimitError as e:
                raise RateLimitError(str(e)) from e
            except openai.AuthenticationError as e:
                raise AuthenticationError(str(e)) from e
            except openai.APIError as e:
                raise ProviderError(str(e)) from e

        return retry_with_backoff(wrapped, max_attempts=self._max_retries)

    @staticmethod
    def _to_openai_messages(messages: list[Message]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    @staticmethod
    def _usage_from_response(response: Any) -> TokenUsage | None:
        usage = response.usage
        if usage is None:
            return None
        return TokenUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate a completion for a single prompt.

        Implemented via :meth:`chat` with a single user message -- Chat
        Completions is OpenAI's one real generation endpoint; there is no
        separate legacy-completions call to prefer.
        """
        return self.chat([Message(role="user", content=prompt)], **kwargs)

    def chat(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        client = self._get_client()
        response = self._call_with_retry(
            lambda: client.chat.completions.create(
                model=self._model,
                messages=self._to_openai_messages(messages),
                **kwargs,
            )
        )
        text = response.choices[0].message.content or ""
        return LLMResponse(text=text, usage=self._usage_from_response(response))

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        client = self._get_client()
        message_stream = self._call_with_retry(
            lambda: client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                **kwargs,
            )
        )
        for chunk in message_stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def model_identity(self) -> ModelIdentity:
        return ModelIdentity(name=self._model, revision=self._model, source="api")

    def provenance(self) -> ComponentProvenance:
        return (
            super()
            .provenance()
            .model_copy(
                update={
                    "config": {"model": self._model, "base_url": self._base_url},
                    "model_identity": self.model_identity(),
                }
            )
        )
