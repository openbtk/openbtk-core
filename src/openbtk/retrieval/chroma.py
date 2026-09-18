"""A Chroma vector store -- embedded (in-memory or on-disk), never a
remote server.

This implementation only wraps Chroma's *embedded* client modes
(``chromadb.Client()``, ephemeral in-memory; ``chromadb.PersistentClient
(path=...)``, on-disk) -- never ``HttpClient``, which talks to a
separately-running (and potentially remote) Chroma server. That is a
real, deliberate scope limit, not an oversight: this class never sends
data offsite as written, and a future ``HttpClient``-backed variant would
need its own ``sends_data_offsite`` declaration and offsite-policy
handling, not a quiet parameter added to this one.

Chroma's own persistence model does not match ``BaseVectorStore``'s
``persist(path)``/``load(path)`` pair: a ``PersistentClient`` writes to
its configured path continuously, with no explicit "save now" step, and
there is no separate "open this on-disk store" call distinct from
constructing the client in the first place. Rather than force an
awkward, misleading ``persist()``/``load()`` implementation onto a model
that does not have one, this class leaves the base class's own
``NotImplementedError`` defaults in place and exposes persistence
through the constructor's ``path`` argument instead -- disclosed here,
not silently worked around.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from openbtk.core._lazy import require
from openbtk.core.base import BaseVectorStore
from openbtk.core.errors import RetrievalError
from openbtk.core.logging import get_logger
from openbtk.core.registry import VECTORSTORE_REGISTRY
from openbtk.core.schemas import SearchResult

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = get_logger(__name__)

_DEFAULT_COLLECTION = "openbtk_default"


@VECTORSTORE_REGISTRY.register("vectorstore.general.chroma")
class ChromaVectorStore(BaseVectorStore):
    """Store and query vectors with an embedded Chroma collection.

    Args:
        path: Directory for on-disk persistence via
            ``chromadb.PersistentClient``. ``None`` (the default) uses an
            ephemeral in-memory client -- gone once the process exits,
            same trade-off as :class:`~openbtk.retrieval.faiss.FAISSVectorStore`
            with no ``persist()`` call.
        collection_name: Chroma requires 3-512 characters from
            ``[a-zA-Z0-9._-]``; the default satisfies that on its own.

    No I/O happens in ``__init__`` (docs/09_CODING_STANDARDS.md rule 11):
    the ``chromadb`` import and the client/collection it builds are
    deferred to first real use.
    """

    def __init__(
        self, *, path: str | None = None, collection_name: str = _DEFAULT_COLLECTION
    ) -> None:
        self._path = path
        self._collection_name = collection_name
        self._collection: Any = None

    def _get_collection(self) -> Any:
        if self._collection is None:
            chromadb = require("chromadb", extra="retrieval")
            client = (
                chromadb.PersistentClient(path=self._path)
                if self._path is not None
                else chromadb.Client()
            )
            self._collection = client.get_or_create_collection(
                name=self._collection_name
            )
        return self._collection

    def upsert(
        self,
        ids: list[str],
        vectors: NDArray[np.float32],
        metadata: list[dict[str, Any]],
    ) -> None:
        collection = self._get_collection()
        # Chroma rejects an empty {} metadata dict outright ("Expected
        # metadata to be a non-empty dict") but accepts None for "no
        # metadata" -- confirmed directly against the real client, not
        # assumed from the base interface's own dict[str, Any] shape,
        # which allows {} (and every other store here does too).
        metadatas = [m if m else None for m in metadata]
        try:
            collection.upsert(
                ids=ids,
                embeddings=np.asarray(vectors, dtype=np.float32).tolist(),
                metadatas=metadatas,
            )
        except Exception as e:
            raise RetrievalError(f"Chroma upsert failed: {e}") from e

    def query(
        self,
        vector: NDArray[np.float32],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        collection = self._get_collection()
        try:
            response = collection.query(
                query_embeddings=[np.asarray(vector, dtype=np.float32).tolist()],
                n_results=top_k,
                where=filter,
            )
        except Exception as e:
            raise RetrievalError(f"Chroma query failed: {e}") from e
        ids = response["ids"][0]
        distances = response["distances"][0]
        metadatas = response["metadatas"][0]
        return [
            SearchResult(id=i, score=-float(d), metadata=m or {})
            for i, d, m in zip(ids, distances, metadatas, strict=True)
        ]

    def delete(self, ids: list[str]) -> None:
        collection = self._get_collection()
        try:
            collection.delete(ids=ids)
        except Exception as e:
            raise RetrievalError(f"Chroma delete failed: {e}") from e
