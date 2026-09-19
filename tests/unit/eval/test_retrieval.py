"""Unit tests for openbtk.eval.retrieval -- hand-computed expected values,
not values read back from the implementation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from openbtk.core.base import BaseEmbeddingProvider, BaseVectorStore
from openbtk.core.schemas import SearchResult, SourceRef
from openbtk.eval.retrieval import (
    RetrievalQuery,
    evaluate_retrieval,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    retriever_from,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TestRecallAtK:
    def test_fraction_of_relevant_found_in_top_k(self) -> None:
        assert recall_at_k(["a", "b", "c"], {"a", "c", "z"}, k=2) == pytest.approx(
            1 / 3
        )

    def test_all_found(self) -> None:
        assert recall_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0

    def test_none_found(self) -> None:
        assert recall_at_k(["x", "y"], {"a"}, k=2) == 0.0

    def test_k_larger_than_the_list(self) -> None:
        assert recall_at_k(["a"], {"a"}, k=10) == 1.0

    def test_duplicates_do_not_consume_rank_slots(self) -> None:
        # de-duplicated ranking is [x, a]; a is inside top-2.
        assert recall_at_k(["x", "x", "a"], {"a"}, k=2) == 1.0

    def test_no_relevant_documents_is_undefined(self) -> None:
        with pytest.raises(ValueError, match="undefined"):
            recall_at_k(["a"], set(), k=1)

    def test_k_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="k must be"):
            recall_at_k(["a"], {"a"}, k=0)


class TestReciprocalRank:
    def test_first_position(self) -> None:
        assert reciprocal_rank(["a", "b"], {"a"}) == 1.0

    def test_third_position(self) -> None:
        assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_uses_the_first_relevant_of_several(self) -> None:
        assert reciprocal_rank(["x", "b", "a"], {"a", "b"}) == 0.5

    def test_not_retrieved_is_zero(self) -> None:
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


class TestNDCG:
    def test_perfect_ranking_is_one(self) -> None:
        assert ndcg_at_k(["a", "b"], {"a": 1.0, "b": 1.0}, k=2) == pytest.approx(1.0)

    def test_binary_relevant_at_rank_two(self) -> None:
        # DCG = 1/log2(3); IDCG = 1/log2(2) = 1
        assert ndcg_at_k(["x", "a"], {"a": 1.0}, k=2) == pytest.approx(1 / math.log2(3))

    def test_graded_gains_hand_computed(self) -> None:
        rel = {"a": 3.0, "b": 2.0, "c": 1.0}
        dcg = 2 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)
        idcg = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
        assert ndcg_at_k(["b", "a", "c"], rel, k=3) == pytest.approx(dcg / idcg)

    def test_k_truncates_both_dcg_and_ideal(self) -> None:
        rel = {"a": 1.0, "b": 1.0}
        assert ndcg_at_k(["x", "a", "b"], rel, k=1) == 0.0

    def test_unjudged_documents_have_zero_gain(self) -> None:
        assert ndcg_at_k(["u1", "u2"], {"a": 1.0}, k=2) == 0.0

    def test_no_positive_judgement_is_undefined(self) -> None:
        with pytest.raises(ValueError, match="undefined"):
            ndcg_at_k(["a"], {"a": 0.0}, k=1)

    def test_duplicates_do_not_inflate_dcg(self) -> None:
        assert ndcg_at_k(["a", "a", "a"], {"a": 1.0}, k=3) == pytest.approx(1.0)

    @given(
        rels=st.dictionaries(
            st.sampled_from("abcdef"), st.floats(0.5, 5.0), min_size=1, max_size=6
        ),
        perm_seed=st.integers(0, 1000),
        k=st.integers(1, 8),
    )
    def test_bounded_in_unit_interval_and_ideal_ordering_is_one(
        self, rels: dict[str, float], perm_seed: int, k: int
    ) -> None:
        shuffled = sorted(rels)
        np.random.default_rng(perm_seed).shuffle(shuffled)
        assert 0.0 <= ndcg_at_k(shuffled, rels, k) <= 1.0 + 1e-12
        ideal = sorted(rels, key=lambda d: -rels[d])
        assert ndcg_at_k(ideal, rels, k) == pytest.approx(1.0)

    @given(
        ranked=st.permutations(list("abcdef")),
        relevant=st.sets(st.sampled_from("abcdef"), min_size=1),
    )
    def test_recall_is_monotone_in_k(
        self, ranked: list[str], relevant: set[str]
    ) -> None:
        values = [recall_at_k(ranked, relevant, k) for k in range(1, 7)]
        assert values == sorted(values)
        assert values[-1] == 1.0


class TestRetrievalQuery:
    def test_relevant_ids_are_the_positive_gains_sorted(self) -> None:
        q = RetrievalQuery(
            query_id="q", query="x", relevance={"b": 1.0, "a": 2.0, "z": 0.0}
        )
        assert q.relevant_ids == ["a", "b"]

    def test_rejects_a_query_with_nothing_relevant(self) -> None:
        with pytest.raises(ValidationError, match="gain > 0"):
            RetrievalQuery(query_id="q", query="x", relevance={"a": 0.0})

    def test_rejects_an_empty_judgement_set(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalQuery(query_id="q", query="x", relevance={})


def _queries() -> list[RetrievalQuery]:
    return [
        RetrievalQuery(query_id="q1", query="one", relevance={"a": 1.0}),
        RetrievalQuery(query_id="q2", query="two", relevance={"b": 1.0, "c": 1.0}),
    ]


class TestEvaluateRetrieval:
    def test_macro_averages_match_hand_computation(self) -> None:
        rankings = {"one": ["a", "x"], "two": ["x", "c", "b"]}
        report = evaluate_retrieval(lambda q: rankings[q], _queries(), ks=(1, 3))
        # q1: recall@1=1, recall@3=1, rr=1.  q2: recall@1=0, recall@3=1, rr=1/2.
        assert report.n_queries == 2
        assert report.recall_at_k[1] == pytest.approx(0.5)
        assert report.recall_at_k[3] == pytest.approx(1.0)
        assert report.mrr == pytest.approx(0.75)
        # q2 nDCG@3: DCG = 1/log2(3) + 1/log2(4); IDCG = 1 + 1/log2(3)
        q2 = (1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3))
        assert report.ndcg_at_k[3] == pytest.approx((1.0 + q2) / 2)

    def test_queries_are_consumed_once_as_a_stream(self) -> None:
        def stream() -> Any:
            yield from _queries()

        report = evaluate_retrieval(lambda q: ["a"], stream(), ks=(1,))
        assert report.n_queries == 2

    def test_empty_queries_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="no queries"):
            evaluate_retrieval(lambda q: [], [], ks=(1,))

    def test_empty_ks_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="ks must not be empty"):
            evaluate_retrieval(lambda q: [], _queries(), ks=())

    def test_bad_k_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="k must be"):
            evaluate_retrieval(lambda q: [], _queries(), ks=(0,))


class _HashEmbedding(BaseEmbeddingProvider):
    """Deterministic 4-d embedding: a one-hot on the text's first letter."""

    sends_data_offsite = False

    @property
    def dimension(self) -> int:
        return 4

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        out = np.zeros((len(texts), 4), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, (ord(t[0]) - ord("a")) % 4] = 1.0
        return out


class _BruteForceStore(BaseVectorStore):
    def __init__(self) -> None:
        self.ids: list[str] = []
        self.vectors = np.zeros((0, 4), dtype=np.float32)
        self.record_ids: dict[str, str] = {}

    def upsert(
        self,
        ids: list[str],
        vectors: NDArray[np.float32],
        metadata: list[dict[str, Any]],
    ) -> None:
        self.ids += ids
        self.vectors = np.vstack([self.vectors, vectors])
        for i, m in zip(ids, metadata, strict=True):
            self.record_ids[i] = m["record_id"]

    def query(
        self,
        vector: NDArray[np.float32],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        scores = self.vectors @ vector
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            SearchResult(
                id=self.ids[i],
                score=float(scores[i]),
                source=SourceRef(record_id=self.record_ids[self.ids[i]]),
            )
            for i in order
        ]

    def delete(self, ids: list[str]) -> None:  # pragma: no cover - unused here
        raise NotImplementedError


class TestRetrieverFrom:
    def _store(self) -> _BruteForceStore:
        store = _BruteForceStore()
        e = _HashEmbedding()
        store.upsert(
            ["c-a", "c-b"],
            e.embed(["apple", "banana"]),
            [{"record_id": "rec-1"}, {"record_id": "rec-2"}],
        )
        return store

    def test_ranks_by_the_real_embedding_and_store(self) -> None:
        retrieve = retriever_from(_HashEmbedding(), self._store(), top_k=2)
        assert retrieve("avocado")[0] == "c-a"
        assert retrieve("berry")[0] == "c-b"

    def test_end_to_end_metrics_through_the_real_components(self) -> None:
        retrieve = retriever_from(_HashEmbedding(), self._store(), top_k=2)
        qs = [
            RetrievalQuery(query_id="1", query="avocado", relevance={"c-a": 1.0}),
            RetrievalQuery(query_id="2", query="berry", relevance={"c-b": 1.0}),
        ]
        report = evaluate_retrieval(retrieve, qs, ks=(1,))
        assert report.mrr == 1.0
        assert report.recall_at_k[1] == 1.0

    def test_id_of_allows_record_level_judging(self) -> None:
        retrieve = retriever_from(
            _HashEmbedding(),
            self._store(),
            top_k=1,
            id_of=lambda r: r.source.record_id if r.source else r.id,
        )
        assert retrieve("avocado") == ["rec-1"]
