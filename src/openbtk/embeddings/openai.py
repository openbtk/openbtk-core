"""OpenAI embedding provider -- a thin adapter over the official ``openai`` SDK.

Same shape as :class:`openbtk.llms.openai.OpenAIProvider`: the SDK does
all the real work (HTTP, auth), this module translates its
request/response/exception shapes to OpenBTK's, and applies
:func:`openbtk.core.retry.retry_with_backoff` around the call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from openbtk.core._lazy import require
from openbtk.core.base import BaseEmbeddingProvider
from openbtk.core.errors import (
    AuthenticationError,
    ConfigError,
    ProviderError,
    RateLimitError,
)
from openbtk.core.logging import get_logger
from openbtk.core.provenance import ComponentProvenance, ModelIdentity
from openbtk.core.registry import EMBEDDING_REGISTRY
from openbtk.core.retry import retry_with_backoff

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

log = get_logger(__name__)

_DEFAULT_MODEL = "text-embedding-3-small"

# Real, documented output widths (platform.openai.com/docs/guides/embeddings)
# for OpenAI's current embedding models -- not derived from a live call,
# same reasoning as HuggingFaceEmbeddingProvider's required `dimension`:
# this provider's dimension property must answer without a real API call.
_KNOWN_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


@EMBEDDING_REGISTRY.register("embedding.general.openai")
class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Embed text via OpenAI's Embeddings API.

    Args:
        model: The OpenAI embedding model id.
        dimension: The model's output width. Defaults to the known value
            for ``model`` (see ``_KNOWN_DIMENSIONS``); required explicitly
            for any other model id (e.g. a future release, or
            ``text-embedding-3-*``'s own optional ``dimensions`` request
            parameter truncating the native width -- pass both
            consistently if you use that).
        api_key: Forwarded to the SDK client. ``None`` lets the SDK
            resolve ``OPENAI_API_KEY`` from the environment.
        base_url: Forwarded to the SDK client. ``None`` uses OpenAI's own API.
        max_retries: Forwarded to :func:`retry_with_backoff` as ``max_attempts``.

    Raises:
        ConfigError: If ``model`` is not a known model and ``dimension``
            was not given explicitly.

    No I/O happens in ``__init__``: the SDK client is constructed lazily,
    on first real call.
    """

    sends_data_offsite: ClassVar[bool] = True

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        dimension: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 5,
    ) -> None:
        if dimension is None:
            dimension = _KNOWN_DIMENSIONS.get(model)
            if dimension is None:
                raise ConfigError(
                    f"Unknown OpenAI embedding model {model!r} -- pass "
                    "dimension= explicitly.",
                    context={"model": model},
                )
        self._model = model
        self._dimension_value = dimension
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

    @property
    def dimension(self) -> int:
        return self._dimension_value

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        client = self._get_client()
        response = self._call_with_retry(
            lambda: client.embeddings.create(model=self._model, input=texts)
        )
        vectors: NDArray[np.float32] = np.array(
            [item.embedding for item in response.data], dtype=np.float32
        )
        return vectors

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
                        "dimension": self._dimension_value,
                    },
                    "model_identity": self.model_identity(),
                }
            )
        )
