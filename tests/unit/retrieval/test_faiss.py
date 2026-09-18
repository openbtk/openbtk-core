"""Unit tests for openbtk.retrieval.faiss.FAISSVectorStore.

Unlike the LLM/embedding SDK tests, these exercise the REAL ``faiss``
library directly rather than mocking it: FAISS is pure local computation
(no network, no cost, fast) and the whole point of testing this class is
verifying its id-mapping/scoring/filtering logic actually works against
real FAISS behaviour, not a guess at its API shape. Gated on ``faiss-cpu``
actually being installed (the ``retrieval`` extra), the same pattern
already used for pandas-dependent tests elsewhere in this project.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from openbtk.core.errors import RetrievalError
from openbtk.core.schemas import SearchResult

_HAS_FAISS = importlib.util.find_spec("faiss") is not None

pytestmark = pytest.mark.skipif(
    not _HAS_FAISS, reason="requires the 'retrieval' extra (faiss-cpu)"
)

if _HAS_FAISS:
    from openbtk.retrieval.faiss import FAISSVectorStore


def _vectors(*rows: list[float]) -> np.ndarray:
    return np.array(rows, dtype=np.float32)


class TestUpsertAndQuery:
    def test_query_finds_an_exact_match_first(self) -> None:
        store = FAISSVectorStore(dimension=3)
        store.upsert(
            ["a", "b", "c"],
            _vectors([1, 0, 0], [0, 1, 0], [0, 0, 1]),
            [{"label": "x"}, {"label": "y"}, {"label": "z"}],
        )
        results = store.query(np.array([1, 0, 0], dtype=np.float32), top_k=1)
        assert len(results) == 1
        assert results[0].id == "a"
        assert isinstance(results[0], SearchResult)

    def test_higher_score_means_more_relevant_for_l2(self) -> None:
        store = FAISSVectorStore(dimension=3, metric="l2")
        store.upsert(
            ["near", "far"],
            _vectors([1, 0, 0], [0, 0, 1]),
            [{}, {}],
        )
        results = store.query(np.array([1, 0, 0], dtype=np.float32), top_k=2)
        by_id = {r.id: r.score for r in results}
        assert by_id["near"] > by_id["far"]

    def test_ip_metric_reports_raw_inner_product_as_the_score(self) -> None:
        store = FAISSVectorStore(dimension=3, metric="ip")
        store.upsert(["a"], _vectors([2, 0, 0]), [{}])
        results = store.query(np.array([1, 0, 0], dtype=np.float32), top_k=1)
        assert results[0].score == pytest.approx(2.0)

    def test_metadata_is_returned_with_results(self) -> None:
        store = FAISSVectorStore(dimension=3)
        store.upsert(["a"], _vectors([1, 0, 0]), [{"section": "Plan"}])
        results = store.query(np.array([1, 0, 0], dtype=np.float32), top_k=1)
        assert results[0].metadata == {"section": "Plan"}

    def test_upsert_on_an_existing_id_overwrites_it(self) -> None:
        store = FAISSVectorStore(dimension=3)
        store.upsert(["a"], _vectors([1, 0, 0]), [{"v": 1}])
        store.upsert(["a"], _vectors([0, 0, 1]), [{"v": 2}])
        results = store.query(np.array([0, 0, 1], dtype=np.float32), top_k=5)
        assert len(results) == 1
        assert results[0].metadata == {"v": 2}

    def test_query_on_an_empty_store_returns_nothing(self) -> None:
        store = FAISSVectorStore(dimension=3)
        results = store.query(np.array([1, 0, 0], dtype=np.float32), top_k=5)
        assert results == []

    def test_top_k_limits_the_result_count(self) -> None:
        store = FAISSVectorStore(dimension=3)
        store.upsert(
            ["a", "b", "c"],
            _vectors([1, 0, 0], [0, 1, 0], [0, 0, 1]),
            [{}, {}, {}],
        )
        results = store.query(np.array([1, 0, 0], dtype=np.float32), top_k=2)
        assert len(results) == 2

    def test_a_top_k_larger_than_the_index_does_not_return_padding_entries(
        self,
    ) -> None:
        """FAISS pads short results with id -1 -- must never leak through
        as a real result."""
        store = FAISSVectorStore(dimension=3)
        store.upsert(["a"], _vectors([1, 0, 0]), [{}])
        results = store.query(np.array([1, 0, 0], dtype=np.float32), top_k=5)
        assert len(results) == 1


class TestFilter:
    def test_filter_keeps_only_matching_metadata(self) -> None:
        store = FAISSVectorStore(dimension=3)
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
        store = FAISSVectorStore(dimension=3)
        store.upsert(["a"], _vectors([1, 0, 0]), [{"label": "x"}])
        results = store.query(
            np.array([1, 0, 0], dtype=np.float32), top_k=5, filter={"label": "nope"}
        )
        assert results == []


class TestDelete:
    def test_deleted_ids_are_absent_from_subsequent_queries(self) -> None:
        store = FAISSVectorStore(dimension=3)
        store.upsert(["a", "b"], _vectors([1, 0, 0], [0, 1, 0]), [{}, {}])
        store.delete(["a"])
        results = store.query(np.array([1, 0, 0], dtype=np.float32), top_k=5)
        assert [r.id for r in results] == ["b"]

    def test_deleting_an_unknown_id_does_not_raise(self) -> None:
        store = FAISSVectorStore(dimension=3)
        store.delete(["never-existed"])  # must not raise


class TestPersistAndLoad:
    def test_round_trips_vectors_metadata_and_ids(self, tmp_path: Path) -> None:
        path = str(tmp_path / "store.faiss")
        store = FAISSVectorStore(dimension=3, metric="ip")
        store.upsert(["a", "b"], _vectors([1, 0, 0], [0, 1, 0]), [{"x": 1}, {"x": 2}])
        store.persist(path)

        loaded = FAISSVectorStore.load(path)
        results = loaded.query(np.array([1, 0, 0], dtype=np.float32), top_k=5)
        by_id = {r.id: r.metadata for r in results}
        assert by_id == {"a": {"x": 1}, "b": {"x": 2}}

    def test_loaded_store_supports_further_upserts_and_deletes(
        self, tmp_path: Path
    ) -> None:
        path = str(tmp_path / "store.faiss")
        store = FAISSVectorStore(dimension=3)
        store.upsert(["a"], _vectors([1, 0, 0]), [{}])
        store.persist(path)

        loaded = FAISSVectorStore.load(path)
        loaded.upsert(["b"], _vectors([0, 1, 0]), [{}])
        loaded.delete(["a"])
        results = loaded.query(np.array([0, 1, 0], dtype=np.float32), top_k=5)
        assert [r.id for r in results] == ["b"]


class TestErrorHandling:
    def test_a_dimension_mismatched_vector_raises_retrieval_error(self) -> None:
        store = FAISSVectorStore(dimension=3)
        with pytest.raises(RetrievalError):
            store.upsert(["a"], _vectors([1, 0]), [{}])

    def test_a_query_failure_is_wrapped_as_retrieval_error(self) -> None:
        store = FAISSVectorStore(dimension=3)
        store.upsert(["a"], _vectors([1, 0, 0]), [{}])
        index = store._get_index()
        index.search = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
        with pytest.raises(RetrievalError):
            store.query(np.array([1, 0, 0], dtype=np.float32), top_k=1)

    def test_a_delete_failure_is_wrapped_as_retrieval_error(self) -> None:
        store = FAISSVectorStore(dimension=3)
        store.upsert(["a"], _vectors([1, 0, 0]), [{}])
        index = store._get_index()
        index.remove_ids = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
        with pytest.raises(RetrievalError):
            store.delete(["a"])

    def test_a_persist_failure_is_wrapped_as_retrieval_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = FAISSVectorStore(dimension=3)
        store.upsert(["a"], _vectors([1, 0, 0]), [{}])

        class _FakeFaissModule:
            @staticmethod
            def write_index(*args: object, **kwargs: object) -> None:
                raise RuntimeError("boom")

        monkeypatch.setattr(
            "openbtk.retrieval.faiss.require",
            lambda module, extra: _FakeFaissModule(),
        )
        with pytest.raises(RetrievalError):
            store.persist("some/path")

    def test_load_with_a_missing_sidecar_file_raises_retrieval_error(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(RetrievalError):
            FAISSVectorStore.load(str(tmp_path / "does-not-exist"))

    def test_a_read_index_failure_is_wrapped_as_retrieval_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = str(tmp_path / "store.faiss")
        store = FAISSVectorStore(dimension=3)
        store.upsert(["a"], _vectors([1, 0, 0]), [{}])
        store.persist(path)

        class _FakeFaissModule:
            @staticmethod
            def read_index(*args: object, **kwargs: object) -> None:
                raise RuntimeError("boom")

        monkeypatch.setattr(
            "openbtk.retrieval.faiss.require",
            lambda module, extra: _FakeFaissModule(),
        )
        with pytest.raises(RetrievalError):
            FAISSVectorStore.load(path)


class TestLaziness:
    def test_constructing_the_store_does_not_import_faiss(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []
        monkeypatch.setattr(
            "openbtk.retrieval.faiss.require",
            lambda module, extra: calls.append(module),
        )
        FAISSVectorStore(dimension=3)
        assert calls == []


class TestDeclaredAttributes:
    def test_registered_under_the_expected_key(self) -> None:
        assert FAISSVectorStore.registry_key == "vectorstore.general.faiss"
