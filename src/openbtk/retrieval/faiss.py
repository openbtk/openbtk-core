"""A local FAISS vector store -- exact nearest-neighbour search, in-process,
no server, never sends data offsite.

FAISS's own index has no concept of a string id, no metadata, and no
delete-by-string-id -- ``faiss.IndexIDMap`` adds int64 id support (and
real removal via ``remove_ids``) over a base index, so this class layers
its own stable string-id <-> int64-id mapping over that, plus a plain
dict for metadata (FAISS stores neither). ``upsert`` on an id that
already exists removes the old vector first, matching the interface's
"insert or update" contract -- ``IndexIDMap`` has no native upsert.

``query``'s ``filter`` is applied as an exact-match-all post-filter
(``all(metadata.get(k) == v for k, v in filter.items())``) after FAISS's
own search, since the base index has no metadata-aware search at all --
a real, disclosed limitation (matching this project's own convention of
disclosing rather than silently working around a gap): filtering can
return fewer than ``top_k`` results even when more matching vectors
exist elsewhere in the index, since the filter never sees vectors
FAISS's own top-k cutoff already excluded.
"""

from __future__ import annotations

import json
from pathlib import Path
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

Metric = Literal["l2", "ip"]


@VECTORSTORE_REGISTRY.register("vectorstore.general.faiss")
class FAISSVectorStore(BaseVectorStore):
    """Store and query vectors with a local FAISS index.

    Args:
        dimension: Vector width -- fixed for the life of the index, like
            every FAISS index.
        metric: ``"l2"`` (Euclidean distance, smaller is closer) or
            ``"ip"`` (inner product, larger is closer -- the right choice
            for pre-normalised embeddings, where inner product is cosine
            similarity). ``SearchResult.score`` is always reported so that
            *higher means more relevant* regardless of metric: ``"l2"``
            distances are negated, ``"ip"`` scores are used as-is.

    No I/O happens in ``__init__`` (docs/09_CODING_STANDARDS.md rule 11):
    the ``faiss`` import (and the index it builds) is deferred to first
    real use, so constructing this class -- e.g. while validating a
    pipeline config -- never requires ``faiss-cpu`` to be installed
    unless a step actually runs.
    """

    def __init__(self, *, dimension: int, metric: Metric = "l2") -> None:
        self._dimension = dimension
        self._metric = metric
        self._index: Any = None
        self._id_to_int: dict[str, int] = {}
        self._int_to_id: dict[int, str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._next_int_id = 0

    def _get_index(self) -> Any:
        if self._index is None:
            faiss = require("faiss", extra="retrieval")
            base = (
                faiss.IndexFlatIP(self._dimension)
                if self._metric == "ip"
                else faiss.IndexFlatL2(self._dimension)
            )
            self._index = faiss.IndexIDMap(base)
        return self._index

    def upsert(
        self,
        ids: list[str],
        vectors: NDArray[np.float32],
        metadata: list[dict[str, Any]],
    ) -> None:
        index = self._get_index()
        stale = [self._id_to_int[i] for i in ids if i in self._id_to_int]
        if stale:
            index.remove_ids(np.array(stale, dtype=np.int64))
        int_ids = []
        for record_id in ids:
            existing = self._id_to_int.get(record_id)
            if existing is not None:
                int_id = existing
            else:
                int_id = self._next_int_id
                self._next_int_id += 1
                self._id_to_int[record_id] = int_id
                self._int_to_id[int_id] = record_id
            int_ids.append(int_id)
        try:
            index.add_with_ids(
                np.ascontiguousarray(vectors, dtype=np.float32),
                np.array(int_ids, dtype=np.int64),
            )
        except Exception as e:
            raise RetrievalError(f"FAISS upsert failed: {e}") from e
        for record_id, meta in zip(ids, metadata, strict=True):
            self._metadata[record_id] = meta

    def query(
        self,
        vector: NDArray[np.float32],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        index = self._get_index()
        if index.ntotal == 0:
            return []
        # Over-fetch when a filter is set: the post-filter below can only
        # narrow what FAISS already returned, so asking for more than
        # top_k up front is what keeps a filtered query from returning
        # fewer results than it needs to whenever possible.
        fetch_k = min(index.ntotal, top_k * 10 if filter else top_k)
        try:
            distances, indices = index.search(
                np.ascontiguousarray(vector, dtype=np.float32).reshape(1, -1), fetch_k
            )
        except Exception as e:
            raise RetrievalError(f"FAISS query failed: {e}") from e
        results = []
        for int_id, distance in zip(indices[0], distances[0], strict=True):
            if int_id == -1:  # pragma: no cover -- defensive: FAISS pads
                # short results with -1 when asked for more than an index
                # holds, but fetch_k above is already clamped to
                # index.ntotal, so this should never actually trigger;
                # kept as a backstop against relying on that undocumented
                # FAISS behaviour holding across versions/index types.
                continue
            record_id = self._int_to_id[int(int_id)]
            meta = self._metadata.get(record_id, {})
            if filter and not all(meta.get(k) == v for k, v in filter.items()):
                continue
            score = float(distance) if self._metric == "ip" else -float(distance)
            results.append(SearchResult(id=record_id, score=score, metadata=meta))
            if len(results) == top_k:
                break
        return results

    def delete(self, ids: list[str]) -> None:
        int_ids = [self._id_to_int.pop(i) for i in ids if i in self._id_to_int]
        for int_id in int_ids:
            del self._int_to_id[int_id]
        for record_id in ids:
            self._metadata.pop(record_id, None)
        if int_ids:
            try:
                self._get_index().remove_ids(np.array(int_ids, dtype=np.int64))
            except Exception as e:
                raise RetrievalError(f"FAISS delete failed: {e}") from e

    def persist(self, path: str) -> None:
        faiss = require("faiss", extra="retrieval")
        try:
            faiss.write_index(self._get_index(), path)
        except Exception as e:
            raise RetrievalError(f"FAISS persist failed: {e}") from e
        sidecar = {
            "dimension": self._dimension,
            "metric": self._metric,
            "id_to_int": self._id_to_int,
            "metadata": self._metadata,
            "next_int_id": self._next_int_id,
        }
        Path(f"{path}.meta.json").write_text(json.dumps(sidecar), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> FAISSVectorStore:
        faiss = require("faiss", extra="retrieval")
        sidecar_path = Path(f"{path}.meta.json")
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except OSError as e:
            raise RetrievalError(f"FAISS load failed: {e}") from e
        store = cls(dimension=sidecar["dimension"], metric=sidecar["metric"])
        try:
            store._index = faiss.read_index(path)
        except Exception as e:
            raise RetrievalError(f"FAISS load failed: {e}") from e
        store._id_to_int = {k: int(v) for k, v in sidecar["id_to_int"].items()}
        store._int_to_id = {v: k for k, v in store._id_to_int.items()}
        store._metadata = sidecar["metadata"]
        store._next_int_id = sidecar["next_int_id"]
        return store
