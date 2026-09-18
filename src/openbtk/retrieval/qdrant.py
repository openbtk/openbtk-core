"""A Qdrant vector store -- embedded (in-memory or on-disk), never a
remote server or Qdrant Cloud.

This implementation only wraps ``QdrantClient(location=":memory:")`` and
``QdrantClient(path=...)`` -- Qdrant's own embedded modes -- never
``QdrantClient(url=...)``/``host=...``, which talks to a separately-running
(and potentially remote, e.g. Qdrant Cloud) server. Same real, deliberate
scope limit as :mod:`openbtk.retrieval.chroma`'s own docstring: this class
never sends data offsite as written, and a future remote-client variant
would need its own ``sends_data_offsite`` declaration.

Qdrant point ids must be an unsigned integer or a UUID -- never an
arbitrary string, unlike this interface's own ``ids: list[str]``. Every
point's id is instead a UUID deterministically derived from the caller's
string id (``uuid.uuid5``, stable across calls -- the same string always
maps to the same UUID, which is what makes ``upsert`` on an existing id
genuinely overwrite the same point rather than create a duplicate), with
the original string preserved in the point's payload and used to
reconstruct ``SearchResult.id`` on the way back out.

Same persistence-model mismatch as Chroma (an embedded/local ``path=``
client persists continuously, with no explicit "save now"/"open this"
pair) -- ``persist()``/``load()`` are left at the base class's
``NotImplementedError`` defaults; persistence is configured via the
constructor's ``path`` argument instead.

**On-disk ``path=`` access is exclusive, not concurrent** -- Qdrant's own
embedded local mode file-locks the storage directory, confirmed directly:
a second ``QdrantVectorStore`` pointed at a ``path`` another instance
still has open raises a real ``RuntimeError`` from the client itself
("Storage folder ... is already accessed by another instance"), not
something this class works around. One more reason this implementation
is scoped to the embedded modes only -- a real server exists precisely to
serve concurrent clients, which local mode is not built for.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Literal

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

Metric = Literal["cosine", "euclid", "dot"]
_DEFAULT_COLLECTION = "openbtk_default"

# A fixed namespace so the same (collection, record_id) pair always
# derives the same UUID across process restarts -- required for upsert-
# on-existing-id to genuinely overwrite the same Qdrant point, not create
# a duplicate under a freshly-random id every run.
_ID_NAMESPACE = uuid.UUID("d3e1a35d-6e6c-4d8a-9e6b-2a8b6f0f7a41")


def _point_id(record_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, record_id))


@VECTORSTORE_REGISTRY.register("vectorstore.general.qdrant")
class QdrantVectorStore(BaseVectorStore):
    """Store and query vectors with an embedded Qdrant collection.

    Args:
        dimension: Vector width, required up front by Qdrant's own
            ``create_collection``.
        metric: ``"cosine"`` (the default) or ``"dot"`` report
            higher-is-better similarity natively; ``"euclid"`` reports a
            raw distance, negated here so ``SearchResult.score`` means
            the same thing (higher is more relevant) regardless of metric
            -- verified directly against the real client, not assumed
            from Qdrant's docs alone.
        path: Directory for on-disk persistence. ``None`` (the default)
            uses ``":memory:"``.
        collection_name: The Qdrant collection to use.

    No I/O happens in ``__init__`` (docs/09_CODING_STANDARDS.md rule 11):
    the ``qdrant_client`` import and the client/collection it builds are
    deferred to first real use.
    """

    def __init__(
        self,
        *,
        dimension: int,
        metric: Metric = "cosine",
        path: str | None = None,
        collection_name: str = _DEFAULT_COLLECTION,
    ) -> None:
        self._dimension = dimension
        self._metric = metric
        self._path = path
        self._collection_name = collection_name
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            qdrant_client = require("qdrant_client", extra="retrieval")
            models = require("qdrant_client.models", extra="retrieval")
            client = qdrant_client.QdrantClient(
                location=":memory:" if self._path is None else None,
                path=self._path,
            )
            if not client.collection_exists(self._collection_name):
                distance = {
                    "cosine": models.Distance.COSINE,
                    "euclid": models.Distance.EUCLID,
                    "dot": models.Distance.DOT,
                }[self._metric]
                client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=models.VectorParams(
                        size=self._dimension, distance=distance
                    ),
                )
            self._client = client
        return self._client

    def upsert(
        self,
        ids: list[str],
        vectors: NDArray[np.float32],
        metadata: list[dict[str, Any]],
    ) -> None:
        models = require("qdrant_client.models", extra="retrieval")
        client = self._get_client()
        points = [
            models.PointStruct(
                id=_point_id(record_id),
                vector=np.asarray(vec, dtype=np.float32).tolist(),
                payload={**meta, "_openbtk_id": record_id},
            )
            for record_id, vec, meta in zip(ids, vectors, metadata, strict=True)
        ]
        try:
            client.upsert(collection_name=self._collection_name, points=points)
        except Exception as e:
            raise RetrievalError(f"Qdrant upsert failed: {e}") from e

    def query(
        self,
        vector: NDArray[np.float32],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        models = require("qdrant_client.models", extra="retrieval")
        client = self._get_client()
        query_filter = None
        if filter:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(key=k, match=models.MatchValue(value=v))
                    for k, v in filter.items()
                ]
            )
        try:
            response = client.query_points(
                collection_name=self._collection_name,
                query=np.asarray(vector, dtype=np.float32).tolist(),
                limit=top_k,
                query_filter=query_filter,
            )
        except Exception as e:
            raise RetrievalError(f"Qdrant query failed: {e}") from e
        results = []
        for point in response.points:
            payload = point.payload or {}
            score = -point.score if self._metric == "euclid" else point.score
            results.append(
                SearchResult(
                    id=payload["_openbtk_id"],
                    score=float(score),
                    metadata={k: v for k, v in payload.items() if k != "_openbtk_id"},
                )
            )
        return results

    def delete(self, ids: list[str]) -> None:
        client = self._get_client()
        try:
            client.delete(
                collection_name=self._collection_name,
                points_selector=[_point_id(i) for i in ids],
            )
        except Exception as e:
            raise RetrievalError(f"Qdrant delete failed: {e}") from e
