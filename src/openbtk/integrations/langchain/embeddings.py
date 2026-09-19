"""An OpenBTK embedding provider as a ``langchain_core`` ``Embeddings``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.embeddings import Embeddings

from openbtk.core.registry import EMBEDDING_REGISTRY

if TYPE_CHECKING:
    from openbtk.core.base import BaseEmbeddingProvider
    from openbtk.core.config import PolicyConfig


class OpenBTKEmbeddings(Embeddings):
    """Expose a ``BaseEmbeddingProvider`` to LangChain.

    Documents are embedded in slices of the provider's own ``batch_size`` so a
    long list is never sent as one oversized request. The provider's
    ``sends_data_offsite`` flag is unchanged by wrapping: an offsite provider
    stays offsite, and a local-only policy is enforced where the provider is
    constructed -- see :meth:`from_registry`.

    Args:
        provider: The provider to wrap.

    Example:
        >>> import numpy as np
        >>> from openbtk.core.base import BaseEmbeddingProvider
        >>> class Fixed(BaseEmbeddingProvider):
        ...     sends_data_offsite = False
        ...     dimension = 2
        ...     def embed(self, texts):
        ...         return np.ones((len(texts), 2), dtype=np.float32)
        >>> OpenBTKEmbeddings(Fixed()).embed_query("hello")
        [1.0, 1.0]
    """

    def __init__(self, provider: BaseEmbeddingProvider) -> None:
        self._provider = provider

    @classmethod
    def from_registry(
        cls, key: str, *, policy: PolicyConfig | None = None, **params: Any
    ) -> OpenBTKEmbeddings:
        """Build the provider registered under ``key`` and wrap it.

        ``policy`` is forwarded to the registry, which refuses an offsite
        provider unless ``policy.allow_offsite_providers`` is true -- wrapping
        a provider for LangChain does not bypass the local-only guard.

        Raises:
            PolicyError: If the provider sends data offsite and ``policy``
                does not allow it.
        """
        return cls(EMBEDDING_REGISTRY.create(key, policy=policy, **params))

    @property
    def provider(self) -> BaseEmbeddingProvider:
        """The wrapped OpenBTK provider."""
        return self._provider

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        step = max(1, self._provider.batch_size)
        for i in range(0, len(texts), step):
            batch = self._provider.embed(texts[i : i + step])
            vectors.extend(batch.astype("float64").tolist())
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return [float(x) for x in self._provider.embed_one(text)]
