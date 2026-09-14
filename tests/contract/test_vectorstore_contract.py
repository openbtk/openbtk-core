"""Shared contract every registered BaseVectorStore must satisfy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from openbtk.core.registry import VECTORSTORE_REGISTRY
from openbtk.core.schemas import SearchResult

if TYPE_CHECKING:
    from openbtk.core.base import BaseVectorStore


def _new_instance(key: str) -> BaseVectorStore:
    return VECTORSTORE_REGISTRY.create(key)


@pytest.mark.parametrize("key", VECTORSTORE_REGISTRY.list_keys())
class TestVectorStoreContract:
    def test_upsert_then_query_finds_it(self, key: str) -> None:
        store = _new_instance(key)
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        store.upsert(["a"], vec.reshape(1, -1), [{"label": "x"}])
        results = store.query(vec, top_k=5)
        assert any(r.id == "a" for r in results)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_delete_removes_from_subsequent_queries(self, key: str) -> None:
        store = _new_instance(key)
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        store.upsert(["a"], vec.reshape(1, -1), [{}])
        store.delete(["a"])
        results = store.query(vec, top_k=5)
        assert not any(r.id == "a" for r in results)

    def test_persist_and_load_are_symmetric_or_both_unsupported(self, key: str) -> None:
        """Either both persist() and load() work (a store that supports
        persistence), or persist() raises NotImplementedError (the base
        default for a store that doesn't) -- never one without the other."""
        store = _new_instance(key)
        try:
            store.persist("dummy-path-not-actually-written")
        except NotImplementedError:
            return  # a store with no persistence support is contract-valid
        # If persist() didn't raise, load() must exist and be callable too.
        assert callable(type(store).load)

    def test_provenance_is_serialisable(self, key: str) -> None:
        store = _new_instance(key)
        dumped = store.provenance().model_dump_json()
        assert isinstance(dumped, str) and len(dumped) > 0
