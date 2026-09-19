"""Small real implementations of OpenBTK's base classes for the adapter tests.
They are working components (a one-hot embedding, a brute-force store, an
echoing LLM), not mocks: the adapters are exercised end to end through them."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from openbtk.core.base import BaseEmbeddingProvider, BaseLLMProvider, BaseVectorStore
from openbtk.core.schemas import LLMResponse, SearchResult, TokenUsage

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

    from openbtk.core.schemas import Message


class OneHotEmbedding(BaseEmbeddingProvider):
    """4-d one-hot on the first letter; records every ``embed`` call's size."""

    sends_data_offsite = False

    def __init__(self, *, batch: int = 32) -> None:
        self._batch = batch
        self.call_sizes: list[int] = []

    @property
    def dimension(self) -> int:
        return 4

    @property
    def batch_size(self) -> int:
        return self._batch

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        self.call_sizes.append(len(texts))
        out = np.zeros((len(texts), 4), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, (ord(t[0].lower()) - ord("a")) % 4] = 1.0
        return out


class BruteForceStore(BaseVectorStore):
    """Exact inner-product search; honours an equality ``filter``."""

    def __init__(self) -> None:
        self.rows: dict[str, tuple[NDArray[np.float32], dict[str, Any]]] = {}
        self.last_filter: dict[str, Any] | None = None

    def upsert(
        self,
        ids: list[str],
        vectors: NDArray[np.float32],
        metadata: list[dict[str, Any]],
    ) -> None:
        for i, v, m in zip(ids, vectors, metadata, strict=True):
            self.rows[i] = (v, m)

    def query(
        self,
        vector: NDArray[np.float32],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        self.last_filter = filter
        scored = [
            (float(v @ vector), i, m)
            for i, (v, m) in self.rows.items()
            if not filter or all(m.get(k) == want for k, want in filter.items())
        ]
        scored.sort(key=lambda t: -t[0])
        return [SearchResult(id=i, score=s, metadata=m) for s, i, m in scored[:top_k]]

    def delete(self, ids: list[str]) -> None:
        for i in ids:
            self.rows.pop(i, None)


class EchoLLM(BaseLLMProvider):
    """Upper-cases the last message; records what it was sent."""

    sends_data_offsite = False

    def __init__(self) -> None:
        self.seen: list[Message] = []
        self.kwargs: dict[str, Any] = {}

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        return LLMResponse(text=prompt.upper())

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        yield prompt.upper()

    def chat(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        self.seen = messages
        self.kwargs = kwargs
        return LLMResponse(
            text=messages[-1].content.upper(),
            usage=TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        )
