"""Unit tests for openbtk.retrieval.chroma.ChromaVectorStore.

Same rationale as test_faiss.py: exercises the REAL ``chromadb`` library
(its ephemeral in-memory client -- no server, no network) rather than
mocking it. Gated on ``chromadb`` actually being installed.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import numpy as np
import pytest

from openbtk.core.errors import RetrievalError
from openbtk.core.schemas import SearchResult

if TYPE_CHECKING:
    from pathlib import Path

_HAS_CHROMADB = importlib.util.find_spec("chromadb") is not None

pytestmark = pytest.mark.skipif(
    not _HAS_CHROMADB, reason="requires the 'retrieval' extra (chromadb)"
)

if _HAS_CHROMADB:
    from openbtk.retrieval.chroma import ChromaVectorStore


def _vectors(*rows: list[float]) -> np.ndarray:
    return np.array(rows, dtype=np.float32)


def _store(**kwargs: object) -> ChromaVectorStore:
    # A fresh, uniquely-named collection per test -- Chroma's in-memory
    # client is process-wide, so reusing a name would leak state between
    # tests otherwise.
    import uuid

    kwargs.setdefault("collection_name", f"test_{uuid.uuid4().hex}")
    return ChromaVectorStore(**kwargs)  # type: ignore[arg-type]


class TestUpsertAndQuery:
    def test_query_finds_an_exact_match_first(self) -> None:
        store = _store()
        store.upsert(
            ["a", "b", "c"],
            _vectors([1, 0, 0], [0, 1, 0], [0, 0, 1]),
            [{"label": "x"}, {"label": "y"}, {"label": "z"}],
        )
        results = store.query(np.array([1, 0, 0], dtype=np.float32), top_k=1)
        assert len(results) == 1
        assert results[0].id == "a"
        assert isinstance(results[0], SearchResult)

    def test_higher_score_means_more_relevant(self) -> None:
        store = _store()
        store.upsert(["near", "far"], _vectors([1, 0, 0], [0, 0, 1]), [{}, {}])
        results = store.query(np.array([1, 0, 0], dtype=np.float32), top_k=2)
        by_id = {r.id: r.score for r in results}
        assert by_id["near"] > by_id["far"]

    def test_metadata_is_returned_with_results(self) -> None:
        store = _store()
        store.upsert(["a"], _vectors([1, 0, 0]), [{"section": "Plan"}])
        results = store.query(np.array([1, 0, 0], dtype=np.float32), top_k=1)
        assert results[0].metadata == {"section": "Plan"}

    def test_upsert_on_an_existing_id_overwrites_it(self) -> None:
        store = _store()
        store.upsert(["a"], _vectors([1, 0, 0]), [{"v": 1}])
        store.upsert(["a"], _vectors([0, 0, 1]), [{"v": 2}])
        results = store.query(np.array([0, 0, 1], dtype=np.float32), top_k=5)
        assert len(results) == 1
        assert results[0].metadata == {"v": 2}

    def test_top_k_limits_the_result_count(self) -> None:
        store = _store()
        store.upsert(
            ["a", "b", "c"],
            _vectors([1, 0, 0], [0, 1, 0], [0, 0, 1]),
            [{}, {}, {}],
        )
        results = store.query(np.array([1, 0, 0], dtype=np.float32), top_k=2)
        assert len(results) == 2


class TestFilter:
    def test_filter_keeps_only_matching_metadata(self) -> None:
        store = _store()
        store.upsert(
            ["a", "b"],
            _vectors([1, 0, 0], [1, 0, 0]),
            [{"label": "x"}, {"label": "y"}],
        )
        results = store.query(
            np.array([1, 0, 0], dtype=np.float32), top_k=5, filter={"label": "y"}
        )
        assert [r.id for r in results] == ["b"]

    def test_filter_matching_nothing_returns_an_empty_list(self) -> None:
        store = _store()
        store.upsert(["a"], _vectors([1, 0, 0]), [{"label": "x"}])
        results = store.query(
            np.array([1, 0, 0], dtype=np.float32), top_k=5, filter={"label": "nope"}
        )
        assert results == []


class TestDelete:
    def test_deleted_ids_are_absent_from_subsequent_queries(self) -> None:
        store = _store()
        store.upsert(["a", "b"], _vectors([1, 0, 0], [0, 1, 0]), [{}, {}])
        store.delete(["a"])
        results = store.query(np.array([1, 0, 0], dtype=np.float32), top_k=5)
        assert [r.id for r in results] == ["b"]


class TestPersistence:
    def test_persistent_client_survives_a_fresh_instance_at_the_same_path(
        self, tmp_path: Path
    ) -> None:
        path = str(tmp_path)
        store = _store(path=path, collection_name="persisted_collection")
        store.upsert(["a"], _vectors([1, 0, 0]), [{"v": 1}])

        reopened = ChromaVectorStore(path=path, collection_name="persisted_collection")
        results = reopened.query(np.array([1, 0, 0], dtype=np.float32), top_k=5)
        assert [r.id for r in results] == ["a"]

    def test_persist_raises_not_implemented_per_the_base_default(self) -> None:
        store = _store()
        with pytest.raises(NotImplementedError):
            store.persist("some/path")

    def test_load_raises_not_implemented_per_the_base_default(self) -> None:
        with pytest.raises(NotImplementedError):
            ChromaVectorStore.load("some/path")


class TestErrorHandling:
    def test_a_query_failure_is_wrapped_as_retrieval_error(self) -> None:
        store = _store()
        collection = store._get_collection()
        collection.query = lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))
        with pytest.raises(RetrievalError):
            store.query(np.array([1, 0, 0], dtype=np.float32), top_k=1)

    def test_an_upsert_failure_is_wrapped_as_retrieval_error(self) -> None:
        store = _store()
        collection = store._get_collection()
        collection.upsert = lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))
        with pytest.raises(RetrievalError):
            store.upsert(["a"], _vectors([1, 0, 0]), [{}])

    def test_a_delete_failure_is_wrapped_as_retrieval_error(self) -> None:
        store = _store()
        store.upsert(["a"], _vectors([1, 0, 0]), [{}])
        collection = store._get_collection()
        collection.delete = lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))
        with pytest.raises(RetrievalError):
            store.delete(["a"])


class TestLaziness:
    def test_constructing_the_store_does_not_import_chromadb(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []
        monkeypatch.setattr(
            "openbtk.retrieval.chroma.require",
            lambda module, extra: calls.append(module),
        )
        _store()
        assert calls == []


class TestDeclaredAttributes:
    def test_registered_under_the_expected_key(self) -> None:
        assert ChromaVectorStore.registry_key == "vectorstore.general.chroma"
