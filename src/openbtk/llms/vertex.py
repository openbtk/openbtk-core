"""Google Vertex AI LLM provider (FR-V-01), over the ``google-genai`` SDK.

Google's current SDK, ``google-genai``, serves both the Gemini Developer API and Vertex
AI from one client; this provider always uses the **Vertex** side (``vertexai=True``),
so requests go to your Google Cloud project, not to an API-key endpoint. The client,
request and response shapes used here were checked against the installed SDK, not
recalled.

* ``system`` messages become the request's ``system_instruction``; ``user`` messages
  stay ``user`` and ``assistant`` messages become the SDK's ``model`` role.
* Generation settings are ``max_tokens`` (sent as ``max_output_tokens``),
  ``temperature``, ``top_p`` and ``stop`` (a list, sent as ``stop_sequences``). Any
  other keyword is **refused**, not passed through or dropped.
* Google Cloud credentials (Application Default Credentials) and the project and
  location come from the environment unless given; they are never read, held or logged
  here.
* HTTP 429 is retried with back-off; 401 and 403 surface as ``AuthenticationError``;
  everything else as ``ProviderError``. Error text is the service's status and message,
  never the prompt.

**Sends data off the machine** (to Google Cloud), so it is refused unless the policy
allows off-site providers. It is tested against a mocked SDK; it has not been run
against a live Google Cloud project. Needs ``pip install "openbtk[vertex]"``.
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

_CONFIG_KEYS = {
    "max_tokens": "max_output_tokens",
    "temperature": "temperature",
    "top_p": "top_p",
    "stop": "stop_sequences",
}


@LLM_REGISTRY.register("llm.general.vertex")
class VertexAIProvider(BaseLLMProvider):
    """Generate text with a model on Google Vertex AI.

    Args:
        model: The Vertex model id, with its version where the model has one. Pin a
            versioned id: an alias is the vendor's moving target.
        project: The Google Cloud project. ``None`` lets the SDK resolve it.
        location: The region, e.g. ``"us-central1"``. ``None`` lets the SDK resolve it.
        max_retries: Rate-limit retry attempts.

    Example:
        >>> provider = VertexAIProvider(model="model-001")  # nothing is called
        >>> provider.sends_data_offsite
        True
    """

    sends_data_offsite: ClassVar[bool] = True

    def __init__(
        self,
        *,
        model: str,
        project: str | None = None,
        location: str | None = None,
        max_retries: int = 5,
    ) -> None:
        self._model = model
        self._project = project
        self._location = location
        self._max_retries = max_retries
        self._client: Any = None

    # ------------------------------------------------------------------- client

    def _get_client(self) -> Any:
        if self._client is None:
            genai = require("google.genai", extra="vertex")
            self._client = genai.Client(
                vertexai=True, project=self._project, location=self._location
            )
        return self._client

    @staticmethod
    def _translate(errors: Any, error: Exception) -> Exception | None:
        """OpenBTK's exception for a ``google-genai`` API error, or ``None``."""
        if not isinstance(error, errors.APIError):
            return None
        code = getattr(error, "code", None)
        status = getattr(error, "status", None) or ""
        text = (
            f"Vertex AI error {code} {status}: {getattr(error, 'message', '')}".strip()
        )
        if code == 429:
            return RateLimitError(text)
        if code in (401, 403):
            return AuthenticationError(text)
        return ProviderError(text)

    def _call_with_retry(self, func: Callable[[], Any]) -> Any:
        errors = require("google.genai.errors", extra="vertex")

        def wrapped() -> Any:
            try:
                return func()
            except Exception as e:
                translated = self._translate(errors, e)
                if translated is None:
                    raise
                raise translated from e

        return retry_with_backoff(wrapped, max_attempts=self._max_retries)

    # ---------------------------------------------------------------- requests

    def _request(
        self, messages: list[Message], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        unknown = sorted(set(kwargs) - set(_CONFIG_KEYS))
        if unknown:
            raise ProviderError(
                f"Unsupported generation argument(s) for Vertex AI: {unknown}. "
                f"Supported: {sorted(_CONFIG_KEYS)}.",
                context={"provider": "llm.general.vertex"},
            )
        types = require("google.genai.types", extra="vertex")
        config: dict[str, Any] = {
            _CONFIG_KEYS[name]: value
            for name, value in kwargs.items()
            if value is not None
        }
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        if system:
            config["system_instruction"] = system
        contents = [
            types.Content(
                role="model" if m.role == "assistant" else "user",
                parts=[types.Part(text=m.content)],
            )
            for m in messages
            if m.role != "system"
        ]
        return {
            "model": self._model,
            "contents": contents,
            "config": types.GenerateContentConfig(**config) if config else None,
        }

    # --------------------------------------------------------------- generation

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        return self.chat([Message(role="user", content=prompt)], **kwargs)

    def chat(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        client = self._get_client()
        request = self._request(messages, kwargs)
        response = self._call_with_retry(
            lambda: client.models.generate_content(**request)
        )
        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            text=response.text or "",
            usage=None
            if usage is None
            else TokenUsage(
                prompt_tokens=usage.prompt_token_count,
                completion_tokens=usage.candidates_token_count,
                total_tokens=usage.total_token_count,
            ),
        )

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        client = self._get_client()
        request = self._request([Message(role="user", content=prompt)], kwargs)
        chunks = self._call_with_retry(
            lambda: client.models.generate_content_stream(**request)
        )
        errors = require("google.genai.errors", extra="vertex")
        try:
            for chunk in chunks:
                if chunk.text:
                    yield chunk.text
        except Exception as e:  # an error can arrive part-way through the stream
            translated = self._translate(errors, e)
            if translated is None:
                raise
            raise translated from e

    # -------------------------------------------------------------- provenance

    def model_identity(self) -> ModelIdentity:
        return ModelIdentity(name=self._model, revision=self._model, source="vertex")

    def provenance(self) -> ComponentProvenance:
        return (
            super()
            .provenance()
            .model_copy(
                update={
                    "config": {
                        "model": self._model,
                        "project": self._project,
                        "location": self._location,
                    },
                    "model_identity": self.model_identity(),
                }
            )
        )
