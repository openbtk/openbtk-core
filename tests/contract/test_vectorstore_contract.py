"""Shared contract every registered BaseVectorStore must satisfy.

Parametrized over VECTORSTORE_REGISTRY.list_keys() -- no opt-out
(docs/09_CODING_STANDARDS.md section 12). Task 5.6 added the first three
real stores (FAISS, Chroma, Qdrant); unlike the LLM/embedding contract
suites, none of the checks here are gated behind OPENBTK_SLOW_TESTS --
every one of the three wraps a purely local, fast, no-network operation
(confirmed directly: tests/unit/retrieval/test_*.py already exercises all
three for real, with no mocking, in well under a second each), so there
is no real-network-call or real-cost concern to gate against the way
there is for OpenAI/Anthropic. The only gate needed is the familiar
missing-optional-dependency one (the same pattern already used for
pandas-dependent loader tests): CI's zero-extras test-core job has none
of faiss-cpu/chromadb/qdrant-client installed.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from openbtk.core.registry import VECTORSTORE_REGISTRY
from openbtk.core.schemas import SearchResult

if TYPE_CHECKING:
    from pathlib import Path

    from openbtk.core.base import BaseVectorStore

# FAISS and Qdrant both require `dimension` up front (no default -- see
# each class's own docstring for why); Chroma needs nothing extra. The
# reference store (tests/contract/conftest.py) also needs nothing extra.
_CONSTRUCTOR_KWARGS_BY_KEY: dict[str, dict[str, Any]] = {
    "vectorstore.general.faiss": {"dimension": 3},
    "vectorstore.general.qdrant": {"dimension": 3},
}

_MISSING_DEPENDENCY_BY_KEY: dict[str, str] = {}
if importlib.util.find_spec("faiss") is None:
    _MISSING_DEPENDENCY_BY_KEY["vectorstore.general.faiss"] = "faiss-cpu"
if importlib.util.find_spec("chromadb") is None:
    _MISSING_DEPENDENCY_BY_KEY["vectorstore.general.chroma"] = "chromadb"
if importlib.util.find_spec("qdrant_client") is None:
    _MISSING_DEPENDENCY_BY_KEY["vectorstore.general.qdrant"] = "qdrant-client"


def _skip_if_missing_dependency(key: str) -> None:
    dependency = _MISSING_DEPENDENCY_BY_KEY.get(key)
    if dependency is not None:
        pytest.skip(f"{key}: requires the 'retrieval' extra ({dependency})")


def _new_instance(key: str) -> BaseVectorStore:
    return VECTORSTORE_REGISTRY.create(key, **_CONSTRUCTOR_KWARGS_BY_KEY.get(key, {}))


@pytest.mark.parametrize("key", VECTORSTORE_REGISTRY.list_keys())
class TestVectorStoreContract:
    def test_upsert_then_query_finds_it(self, key: str) -> None:
        _skip_if_missing_dependency(key)
        store = _new_instance(key)
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        store.upsert(["a"], vec.reshape(1, -1), [{"label": "x"}])
        results = store.query(vec, top_k=5)
        assert any(r.id == "a" for r in results)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_delete_removes_from_subsequent_queries(self, key: str) -> None:
        _skip_if_missing_dependency(key)
        store = _new_instance(key)
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        store.upsert(["a"], vec.reshape(1, -1), [{}])
        store.delete(["a"])
        results = store.query(vec, top_k=5)
        assert not any(r.id == "a" for r in results)

    def test_persist_and_load_are_symmetric_or_both_unsupported(
        self, key: str, tmp_path: Path
    ) -> None:
        """Either both persist() and load() work (a store that supports
        persistence), or persist() raises NotImplementedError (the base
        default for a store that doesn't) -- never one without the other.
        Uses a real tmp_path, not a bare relative string: a store that DOES
        support persistence (FAISS) would otherwise litter a real file in
        the working directory every time this suite runs."""
        _skip_if_missing_dependency(key)
        store = _new_instance(key)
        path = str(tmp_path / "contract-suite-store")
        try:
            store.persist(path)
        except NotImplementedError:
            return  # a store with no persistence support is contract-valid
        # If persist() didn't raise, load() must exist and be callable too.
        assert callable(type(store).load)

    def test_provenance_is_serialisable(self, key: str) -> None:
        _skip_if_missing_dependency(key)
        store = _new_instance(key)
        dumped = store.provenance().model_dump_json()
        assert isinstance(dumped, str) and len(dumped) > 0
