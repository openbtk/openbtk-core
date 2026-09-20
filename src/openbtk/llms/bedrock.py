"""AWS Bedrock LLM provider (FR-V-01), over the model-agnostic Converse API.

Bedrock hosts models from several vendors behind one API. This adapter uses **Converse**
(and ConverseStream), which takes the same request shape whichever model answers, so one
class serves them all; you name the model by its Bedrock ``model`` id (or an inference
profile id). The request and response shapes used here were read from botocore's own
``bedrock-runtime`` service definition, not recalled.

* ``system`` messages become Converse ``system`` blocks; ``user`` and ``assistant``
  messages become the conversation.
* Generation settings are the four Converse's ``inferenceConfig`` defines:
  ``max_tokens``, ``temperature``, ``top_p`` and ``stop`` (a list). Any other keyword is
  **refused** rather than passed through, because model-specific request fields are not
  portable and silently dropping one would change what you asked for.
* AWS credentials and region come from the standard AWS chain (environment, profile,
  role); they are never read, held or logged here.
* Throttling is retried with back-off; access and credential errors surface as
  ``AuthenticationError``; everything else as ``ProviderError``. Error text is the
  service's error *code* and message, never the prompt.

**Sends data off the machine** (to AWS), so it is refused unless the policy allows
off-site providers. It is tested against a mocked ``boto3``; it has not been run against
a live AWS account. Needs ``pip install "openbtk[bedrock]"``.
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

_THROTTLING = frozenset({"ThrottlingException", "ServiceQuotaExceededException"})
_AUTH = frozenset(
    {
        "AccessDeniedException",
        "UnrecognizedClientException",
        "ExpiredTokenException",
        "InvalidSignatureException",
        "UnauthorizedException",
    }
)
_INFERENCE_KEYS = {
    "max_tokens": "maxTokens",
    "temperature": "temperature",
    "top_p": "topP",
    "stop": "stopSequences",
}


@LLM_REGISTRY.register("llm.general.bedrock")
class BedrockProvider(BaseLLMProvider):
    """Generate text with a model hosted on AWS Bedrock.

    Args:
        model: The Bedrock model id or inference profile id, e.g. a vendor's id with its
            version suffix. Pin a versioned id: an unversioned alias is the vendor's
            moving target.
        region_name: The AWS region. ``None`` lets boto3 resolve it.
        max_retries: Throttling retry attempts.

    Example:
        >>> provider = BedrockProvider(model="vendor.model-v1:0")  # nothing is called
        >>> provider.sends_data_offsite
        True
    """

    sends_data_offsite: ClassVar[bool] = True

    def __init__(
        self, *, model: str, region_name: str | None = None, max_retries: int = 5
    ) -> None:
        self._model = model
        self._region_name = region_name
        self._max_retries = max_retries
        self._client: Any = None

    # ------------------------------------------------------------------- client

    def _get_client(self) -> Any:
        if self._client is None:
            boto3 = require("boto3", extra="bedrock")
            self._client = boto3.client(
                "bedrock-runtime", region_name=self._region_name
            )
        return self._client

    @staticmethod
    def _translate(exceptions: Any, error: Exception) -> Exception | None:
        """OpenBTK's exception for a botocore error, or ``None`` for any other."""
        if isinstance(error, exceptions.ClientError):
            detail = error.response.get("Error", {})
            code = str(detail.get("Code", "Unknown"))
            text = f"Bedrock error {code}: {detail.get('Message', '')}"
            if code in _THROTTLING:
                return RateLimitError(text)
            if code in _AUTH:
                return AuthenticationError(text)
            return ProviderError(text)
        if isinstance(error, exceptions.NoCredentialsError):
            return AuthenticationError("No AWS credentials were found for Bedrock.")
        if isinstance(error, exceptions.BotoCoreError):
            return ProviderError(f"Bedrock request failed: {type(error).__name__}")
        return None

    def _call_with_retry(self, func: Callable[[], Any]) -> Any:
        exceptions = require("botocore.exceptions", extra="bedrock")

        def wrapped() -> Any:
            try:
                return func()
            except Exception as e:
                translated = self._translate(exceptions, e)
                if translated is None:
                    raise
                raise translated from e

        return retry_with_backoff(wrapped, max_attempts=self._max_retries)

    # ---------------------------------------------------------------- requests

    @staticmethod
    def _inference_config(kwargs: dict[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(kwargs) - set(_INFERENCE_KEYS))
        if unknown:
            raise ProviderError(
                f"Unsupported generation argument(s) for Bedrock: {unknown}. "
                f"Supported: {sorted(_INFERENCE_KEYS)}.",
                context={"provider": "llm.general.bedrock"},
            )
        return {
            _INFERENCE_KEYS[name]: value
            for name, value in kwargs.items()
            if value is not None
        }

    def _request(
        self, messages: list[Message], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "modelId": self._model,
            "messages": [
                {"role": m.role, "content": [{"text": m.content}]}
                for m in messages
                if m.role != "system"
            ],
        }
        system = [{"text": m.content} for m in messages if m.role == "system"]
        if system:
            request["system"] = system
        config = self._inference_config(kwargs)
        if config:
            request["inferenceConfig"] = config
        return request

    # --------------------------------------------------------------- generation

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        return self.chat([Message(role="user", content=prompt)], **kwargs)

    def chat(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        client = self._get_client()
        request = self._request(messages, kwargs)
        response = self._call_with_retry(lambda: client.converse(**request))
        blocks = response.get("output", {}).get("message", {}).get("content", [])
        text = "".join(block["text"] for block in blocks if "text" in block)
        usage = response.get("usage")
        return LLMResponse(
            text=text,
            usage=None
            if not usage
            else TokenUsage(
                prompt_tokens=usage.get("inputTokens"),
                completion_tokens=usage.get("outputTokens"),
                total_tokens=usage.get("totalTokens"),
            ),
        )

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        client = self._get_client()
        request = self._request([Message(role="user", content=prompt)], kwargs)
        response = self._call_with_retry(lambda: client.converse_stream(**request))
        exceptions = require("botocore.exceptions", extra="bedrock")
        try:
            for event in response["stream"]:
                delta = event.get("contentBlockDelta", {}).get("delta", {})
                if delta.get("text"):
                    yield delta["text"]
        except Exception as e:  # an error can arrive mid-stream, as an event
            translated = self._translate(exceptions, e)
            if translated is None:
                raise
            raise translated from e

    # -------------------------------------------------------------- provenance

    def model_identity(self) -> ModelIdentity:
        return ModelIdentity(name=self._model, revision=self._model, source="bedrock")

    def provenance(self) -> ComponentProvenance:
        return (
            super()
            .provenance()
            .model_copy(
                update={
                    "config": {"model": self._model, "region_name": self._region_name},
                    "model_identity": self.model_identity(),
                }
            )
        )
