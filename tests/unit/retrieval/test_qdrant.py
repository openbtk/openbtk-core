"""Unit tests for openbtk.retrieval.qdrant.QdrantVectorStore.

Same rationale as test_faiss.py/test_chroma.py: exercises the REAL
``qdrant_client`` library (its ``:memory:`` embedded mode -- no server,
no network) rather than mocking it. Gated on ``qdrant_client`` actually
being installed.
"""

from __future__ import annotations

import importlib.util
import uuid

import numpy as np
import pytest

from openbtk.core.errors import RetrievalError
from openbtk.core.schemas import SearchResult

_HAS_QDRANT = importlib.util.find_spec("qdrant_client") is not None

pytestmark = pytest.mark.skipif(
    not _HAS_QDRANT, reason="requires the 'retrieval' extra (qdrant-client)"
)

if _HAS_QDRANT:
    from openbtk.retrieval.qdrant import QdrantVectorStore


def _vectors(*rows: list[float]) -> np.ndarray:
    return np.array(rows, dtype=np.float32)


def _store(**kwargs: object) -> QdrantVectorStore:
    kwargs.setdefault("dimension", 3)
    kwargs.setdefault("collection_name", f"test_{uuid.uuid4().hex}")
    return QdrantVectorStore(**kwargs)  # type: ignore[arg-type]


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

    def test_cosine_reports_higher_is_better_natively(self) -> None:
        store = _store(metric="cosine")
        store.upsert(["near", "far"], _vectors([1, 0, 0], [0, 0, 1]), [{}, {}])
        results = store.query(np.array([1, 0, 0], dtype=np.float32), top_k=2)
        by_id = {r.id: r.score for r in results}
        assert by_id["near"] > by_id["far"]

    def test_euclid_score_is_negated_so_higher_is_still_better(self) -> None:
        store = _store(metric="euclid")
        store.upsert(["near", "far"], _vectors([1, 0, 0], [0, 0, 1]), [{}, {}])
        results = store.query(np.array([1, 0, 0], dtype=np.float32), top_k=2)
        by_id = {r.id: r.score for r in results}
        assert by_id["near"] > by_id["far"]
        assert by_id["near"] == pytest.approx(0.0)

    def test_dot_metric_reports_raw_inner_product_as_the_score(self) -> None:
        store = _store(metric="dot")
        store.upsert(["a"], _vectors([2, 0, 0]), [{}])
        results = store.query(np.array([1, 0, 0], dtype=np.float32), top_k=1)
        assert results[0].score == pytest.approx(2.0)

    def test_metadata_is_returned_with_results(self) -> None:
        store = _store()
        store.upsert(["a"], _vectors([1, 0, 0]), [{"section": "Plan"}])
        results = store.query(np.array([1, 0, 0], dtype=np.float32), top_k=1)
        assert results[0].metadata == {"section": "Plan"}

    def test_upsert_on_an_existing_id_overwrites_the_same_point(self) -> None:
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


class TestCollectionReuse:
    def test_a_second_store_pointed_at_the_same_collection_does_not_recreate_it(
        self,
    ) -> None:
        """collection_exists() must be honoured, not just called -- a
        second store instance opened *after* the first is done with the
        storage path (the real shape of a later pipeline run reopening
        one already created) must find the existing data, not silently
        start empty. The two cannot be open concurrently -- Qdrant's own
        embedded local mode file-locks the storage path, confirmed
        directly (a real RuntimeError, not assumed): sequential access
        only, by design, one real reason this implementation only wraps
        the embedded modes at all (see this module's own docstring)."""
        path = f"/tmp/openbtk-qdrant-{uuid.uuid4().hex}"
        name = "shared_collection"
        first = QdrantVectorStore(dimension=3, path=path, collection_name=name)
        first.upsert(["a"], _vectors([1, 0, 0]), [{}])
        first._get_client().close()

        second = QdrantVectorStore(dimension=3, path=path, collection_name=name)
        results = second.query(np.array([1, 0, 0], dtype=np.float32), top_k=5)
        assert [r.id for r in results] == ["a"]
        second._get_client().close()


class TestPersistence:
    def test_persist_raises_not_implemented_per_the_base_default(self) -> None:
        store = _store()
        with pytest.raises(NotImplementedError):
            store.persist("some/path")

    def test_load_raises_not_implemented_per_the_base_default(self) -> None:
        with pytest.raises(NotImplementedError):
            QdrantVectorStore.load("some/path")


class TestErrorHandling:
    def test_an_upsert_failure_is_wrapped_as_retrieval_error(self) -> None:
        store = _store()
        client = store._get_client()
        client.upsert = lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))
        with pytest.raises(RetrievalError):
            store.upsert(["a"], _vectors([1, 0, 0]), [{}])

    def test_a_query_failure_is_wrapped_as_retrieval_error(self) -> None:
        store = _store()
        client = store._get_client()
        client.query_points = lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))
        with pytest.raises(RetrievalError):
            store.query(np.array([1, 0, 0], dtype=np.float32), top_k=1)

    def test_a_delete_failure_is_wrapped_as_retrieval_error(self) -> None:
        store = _store()
        store.upsert(["a"], _vectors([1, 0, 0]), [{}])
        client = store._get_client()
        client.delete = lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))
        with pytest.raises(RetrievalError):
            store.delete(["a"])


class TestLaziness:
    def test_constructing_the_store_does_not_import_qdrant_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []
        monkeypatch.setattr(
            "openbtk.retrieval.qdrant.require",
            lambda module, extra: calls.append(module),
        )
        _store()
        assert calls == []


class TestDeclaredAttributes:
    def test_registered_under_the_expected_key(self) -> None:
        assert QdrantVectorStore.registry_key == "vectorstore.general.qdrant"
