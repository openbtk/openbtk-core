"""A provider for any endpoint that speaks the OpenAI Chat Completions wire
format without being OpenAI -- self-hosted vLLM, Ollama, LM Studio, and
most third-party inference providers all implement this exact JSON shape.

Built directly on ``httpx`` (docs/09_CODING_STANDARDS.md's core dependency
list) rather than the ``openai`` SDK: the wire format, not OpenAI's
service, is the actual contract here, and ``httpx`` is already a core
dependency with nothing else in ``src/`` using it yet -- this is its first
real caller. Pulling in the ``openai`` package (an optional extra) just to
talk to something that is not OpenAI would be the wrong dependency for the
job; a caller who explicitly wants the official SDK pointed at a custom
endpoint can already do that via ``OpenAIProvider(base_url=...)``.

``sends_data_offsite`` is conservatively ``True`` at the class level (the
registry's policy gate checks it before construction -- see
``Registry.create``'s own docstring -- so it cannot vary per-instance by
``base_url``): a self-hosted endpoint reachable at ``http://localhost:...``
and a hosted third-party one both construct this same class, and the
registry cannot tell them apart before ``base_url`` is even known. Treating
every instance as offsite is the safe default this project always takes
when it cannot prove otherwise (docs/06_SECURITY_COMPLIANCE.md section
3.3) -- a real, disclosed conservatism, not an oversight.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar

import httpx

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

_DEFAULT_TIMEOUT = 60.0


@LLM_REGISTRY.register("llm.general.openai_compatible")
class OpenAICompatibleProvider(BaseLLMProvider):
    """Generate text from any OpenAI-Chat-Completions-shaped HTTP endpoint.

    Args:
        base_url: The endpoint's base URL, e.g. ``"http://localhost:8000/v1"``.
            Required -- there is no universal default for "some
            OpenAI-compatible endpoint."
        model: The model identifier the endpoint expects, e.g. a local
            vLLM deployment's served model name.
        api_key: Sent as ``Authorization: Bearer <api_key>`` if given.
            Many self-hosted endpoints accept any value or none at all --
            ``None`` (the default) sends no ``Authorization`` header.
        max_retries: Forwarded to :func:`~openbtk.llms.base.retry_with_backoff`
            as ``max_attempts`` for rate-limit retries.
        timeout: Per-request timeout in seconds, forwarded to ``httpx``.

    No I/O happens in ``__init__`` (docs/09_CODING_STANDARDS.md rule 11):
    the ``httpx.Client`` is constructed lazily, on first real call.
    """

    sends_data_offsite: ClassVar[bool] = True

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        max_retries: int = 5,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required and must not be empty.")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._max_retries = max_retries
        self._timeout = timeout
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.Client(
                base_url=self._base_url, headers=headers, timeout=self._timeout
            )
        return self._client

    def _call_with_retry(self, func: Callable[[], Any]) -> Any:
        """Run ``func`` (an HTTP call), translating status/connection
        errors to OpenBTK's exceptions before retry_with_backoff sees them
        -- it only recognises OpenBTK's own ``RateLimitError``."""

        def wrapped() -> Any:
            try:
                response = func()
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status == 429:
                    raise RateLimitError(str(e)) from e
                if status in (401, 403):
                    raise AuthenticationError(str(e)) from e
                raise ProviderError(str(e)) from e
            except httpx.RequestError as e:
                raise ProviderError(str(e)) from e

        return retry_with_backoff(wrapped, max_attempts=self._max_retries)

    @staticmethod
    def _to_wire_messages(messages: list[Message]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    @staticmethod
    def _usage_from_payload(payload: dict[str, Any]) -> TokenUsage | None:
        usage = payload.get("usage")
        if not usage:
            return None
        return TokenUsage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate a completion for a single prompt.

        Implemented via :meth:`chat` with a single user message.
        """
        return self.chat([Message(role="user", content=prompt)], **kwargs)

    def chat(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        client = self._get_client()
        body = {
            "model": self._model,
            "messages": self._to_wire_messages(messages),
            **kwargs,
        }
        response = self._call_with_retry(
            lambda: client.post("/chat/completions", json=body)
        )
        payload = response.json()
        text = payload["choices"][0]["message"]["content"] or ""
        return LLMResponse(text=text, usage=self._usage_from_payload(payload))

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream a completion via server-sent events.

        Only opening the request (until the response headers arrive, where
        an error status would surface) goes through retry -- see
        AnthropicProvider.stream's docstring for why a partially-consumed
        stream is never retried.
        """
        client = self._get_client()
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            **kwargs,
        }
        stream_context = client.stream("POST", "/chat/completions", json=body)
        response = self._call_with_retry(stream_context.__enter__)
        try:
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: ") :]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"].get("content")
                if delta:
                    yield delta
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
                    "config": {"model": self._model, "base_url": self._base_url},
                    "model_identity": self.model_identity(),
                }
            )
        )
