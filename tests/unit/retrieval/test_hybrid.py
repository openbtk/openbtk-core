"""BM25 and reciprocal-rank fusion (FR-R-05). All text is synthetic."""

from __future__ import annotations

import math
import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from openbtk.core.errors import RetrievalError
from openbtk.core.schemas import SearchResult
from openbtk.retrieval.hybrid import (
    BM25Index,
    default_tokenizer,
    reciprocal_rank_fusion,
)


def _index(**kwargs: object) -> BM25Index:
    return BM25Index(**kwargs)  # type: ignore[arg-type]


def _hit(id_: str, score: float = 1.0, **metadata: str) -> SearchResult:
    return SearchResult(id=id_, score=score, metadata=dict(metadata))


class TestTokenizer:
    def test_lowercases_and_splits_on_non_alphanumerics(self) -> None:
        assert default_tokenizer("Metformin 500-mg, BID; LOINC_58410-2") == [
            "metformin",
            "500",
            "mg",
            "bid",
            "loinc",
            "58410",
            "2",
        ]

    def test_empty_and_symbol_only_text(self) -> None:
        assert default_tokenizer("") == [] and default_tokenizer("!!! --- ???") == []

    def test_non_ascii_letters_are_kept(self) -> None:
        assert default_tokenizer("Café über") == ["café", "über"]


class TestScoring:
    def test_a_score_matches_the_hand_computed_bm25_value(self) -> None:
        index = _index()  # k1=1.5, b=0.75
        index.upsert(["d1", "d2", "d3"], ["a b", "a c c", "d"])
        # N=3, df(c)=1, idf=ln(1+(3-1+0.5)/(1+0.5)); d2: tf=2, len=3, avg=(2+3+1)/3=2
        idf = math.log(1 + 2.5 / 1.5)
        norm = 1 - 0.75 + 0.75 * 3 / 2
        expected = idf * 2 * 2.5 / (2 + 1.5 * norm)
        (hit,) = index.search("c")
        assert hit.id == "d2" and hit.score == pytest.approx(expected)

    def test_a_rarer_term_outweighs_a_common_one(self) -> None:
        index = _index()
        index.upsert(
            ["a", "b", "c", "d"],
            [
                "patient with fever",
                "patient with cough",
                "patient with rash",
                "patient with hydroxychloroquine",
            ],
        )
        hits = index.search("patient hydroxychloroquine")
        assert hits[0].id == "d"

    def test_a_term_in_every_document_still_scores_positively(self) -> None:
        index = _index()
        index.upsert(["a", "b"], ["common word", "common thing"])
        hits = index.search("common")
        assert len(hits) == 2 and all(h.score > 0 for h in hits)

    def test_a_shorter_document_outranks_a_longer_one_with_the_same_term_count(
        self,
    ) -> None:
        index = _index()
        index.upsert(["short", "long"], ["aspirin", "aspirin " + "filler " * 30])
        assert [h.id for h in index.search("aspirin")] == ["short", "long"]

    def test_with_no_length_normalisation_length_does_not_matter(self) -> None:
        index = _index(b=0.0)
        index.upsert(["short", "long"], ["aspirin", "aspirin " + "filler " * 30])
        scores = {h.id: h.score for h in index.search("aspirin")}
        assert scores["short"] == pytest.approx(scores["long"])

    def test_with_k1_zero_repetition_does_not_matter(self) -> None:
        index = _index(k1=0.0)
        index.upsert(
            ["once", "thrice", "other"], ["aspirin", "aspirin aspirin aspirin", "x"]
        )
        scores = {h.id: h.score for h in index.search("aspirin")}
        assert scores["once"] == pytest.approx(scores["thrice"])

    def test_repeated_terms_in_the_query_count_once(self) -> None:
        index = _index()
        index.upsert(["a", "b"], ["aspirin", "insulin"])
        assert index.search("aspirin aspirin aspirin")[0].score == pytest.approx(
            index.search("aspirin")[0].score
        )

    def test_a_code_with_punctuation_matches_exactly(self) -> None:
        index = _index()
        index.upsert(["lab", "note"], ["LOINC 58410-2 panel", "no code here"])
        assert [h.id for h in index.search("58410-2")] == ["lab"]

    def test_ties_break_on_id_so_the_order_is_deterministic(self) -> None:
        index = _index()
        index.upsert(["b", "a", "c"], ["same text"] * 3)
        assert [h.id for h in index.search("same")] == ["a", "b", "c"]

    def test_top_k_and_edge_cases(self) -> None:
        index = _index()
        index.upsert(["a", "b", "c"], ["x"] * 3)
        assert len(index.search("x", top_k=2)) == 2
        assert index.search("x", top_k=0) == []
        assert index.search("nothing indexed has this") == []
        assert _index().search("x") == []
        assert index.search("") == []

    def test_a_custom_tokenizer_is_used_for_text_and_queries(self) -> None:
        index = _index(tokenizer=str.split)
        index.upsert(["a"], ["Aspirin-Dose 5mg"])
        assert index.search("aspirin") == []  # case- and punctuation-sensitive here
        assert [h.id for h in index.search("Aspirin-Dose")] == ["a"]


class TestMetadata:
    def test_hits_carry_the_text_under_the_metadata_key(self) -> None:
        index = _index()
        index.upsert(["a"], ["aspirin dose"])
        assert index.search("aspirin")[0].metadata == {"text": "aspirin dose"}

    def test_supplied_metadata_is_kept_and_the_callers_dict_is_not_mutated(
        self,
    ) -> None:
        supplied = {"record_id": "note-1"}
        index = _index()
        index.upsert(["a"], ["aspirin"], [supplied])
        assert index.search("aspirin")[0].metadata == {
            "record_id": "note-1",
            "text": "aspirin",
        }
        assert supplied == {"record_id": "note-1"}

    def test_the_callers_own_text_field_is_not_overwritten(self) -> None:
        index = _index()
        index.upsert(["a"], ["aspirin"], [{"text": "the original chunk"}])
        assert index.search("aspirin")[0].metadata["text"] == "the original chunk"

    def test_a_custom_text_key(self) -> None:
        index = _index(text_metadata_key="body")
        index.upsert(["a"], ["aspirin"])
        assert index.search("aspirin")[0].metadata == {"body": "aspirin"}

    def test_a_hit_cannot_be_used_to_change_the_index(self) -> None:
        index = _index()
        index.upsert(["a"], ["aspirin"])
        index.search("aspirin")[0].metadata["text"] = "tampered"  # a copy
        assert index.search("aspirin")[0].metadata["text"] == "aspirin"


class TestMaintenance:
    def test_upsert_replaces_an_existing_id(self) -> None:
        index = _index()
        index.upsert(["a", "b"], ["aspirin", "insulin"])
        index.upsert(["a"], ["metformin"])
        assert index.search("aspirin") == []
        assert [h.id for h in index.search("metformin")] == ["a"]
        assert len(index) == 2

    def test_delete_removes_a_chunk_and_forgets_its_text(self) -> None:
        index = _index()
        index.upsert(["a", "b"], ["aspirin", "insulin"])
        index.delete(["a", "never-indexed"])
        assert index.search("aspirin") == [] and len(index) == 1
        assert "aspirin" not in repr(index.__dict__["_metadata"])  # nothing retained

    def test_after_churn_the_index_scores_like_a_fresh_one(self) -> None:
        churned = _index()
        churned.upsert(
            ["a", "b", "c"], ["aspirin dose", "insulin dose", "old text here"]
        )
        churned.upsert(["c"], ["aspirin insulin"])
        churned.delete(["b"])
        churned.upsert(["d"], ["dose of aspirin"])
        fresh = _index()
        fresh.upsert(
            ["a", "c", "d"], ["aspirin dose", "aspirin insulin", "dose of aspirin"]
        )
        for query in ("aspirin", "dose", "insulin aspirin", "old"):
            got = [(h.id, round(h.score, 9)) for h in churned.search(query)]
            want = [(h.id, round(h.score, 9)) for h in fresh.search(query)]
            assert got == want

    def test_an_id_can_be_reused_after_deletion(self) -> None:
        index = _index()
        index.upsert(["a"], ["aspirin"])
        index.delete(["a"])
        index.upsert(["a"], ["insulin"])
        assert [h.id for h in index.search("insulin")] == ["a"]

    @pytest.mark.parametrize(
        ("ids", "texts", "metadata"),
        [(["a"], [], None), (["a", "b"], ["x"], None), (["a"], ["x"], [{}, {}])],
    )
    def test_mismatched_lengths_are_refused(
        self, ids: list[str], texts: list[str], metadata: list[dict[str, str]] | None
    ) -> None:
        with pytest.raises(RetrievalError, match="same length"):
            _index().upsert(ids, texts, metadata)  # type: ignore[arg-type]

    def test_duplicate_ids_in_one_call_are_refused(self) -> None:
        with pytest.raises(RetrievalError, match="unique"):
            _index().upsert(["a", "a"], ["x", "y"])

    @pytest.mark.parametrize("kwargs", [{"k1": -1}, {"b": -0.1}, {"b": 1.5}])
    def test_bad_parameters_are_refused(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError, match="k1"):
            _index(**kwargs)


class TestScale:
    def test_a_large_index_answers_quickly(self) -> None:
        index = _index()
        ids = [f"c{i}" for i in range(20000)]
        index.upsert(
            ids,
            [
                f"synthetic note {i} about topic {i % 50} and word{i}"
                for i in range(20000)
            ],
        )
        started = time.perf_counter()
        for i in range(0, 20000, 500):
            assert index.search(f"word{i}")[0].id == f"c{i}"
        assert time.perf_counter() - started < 10.0  # a backstop; not a benchmark


class TestFusion:
    def test_a_hit_in_both_rankings_beats_a_hit_in_one(self) -> None:
        dense = [_hit("a"), _hit("b")]
        sparse = [_hit("b"), _hit("c")]
        assert [h.id for h in reciprocal_rank_fusion([dense, sparse])] == [
            "b",
            "a",
            "c",
        ]

    def test_the_score_is_the_sum_of_reciprocal_ranks(self) -> None:
        fused = reciprocal_rank_fusion([[_hit("a")], [_hit("x"), _hit("a")]], k=60)
        assert fused[0].id == "a"
        assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)

    def test_only_order_matters_not_the_scale_of_the_scores(self) -> None:
        small = reciprocal_rank_fusion(
            [[_hit("a", 0.1), _hit("b", 0.09)], [_hit("b", 0.2)]]
        )
        large = reciprocal_rank_fusion(
            [[_hit("a", 900), _hit("b", 1)], [_hit("b", 5e6)]]
        )
        assert [h.id for h in small] == [h.id for h in large]

    def test_weights_shift_the_balance(self) -> None:
        dense = [_hit("d1"), _hit("d2")]
        sparse = [_hit("s1"), _hit("s2")]
        assert reciprocal_rank_fusion([dense, sparse], weights=[1, 1])[0].id == "d1"
        assert reciprocal_rank_fusion([dense, sparse], weights=[1, 3])[0].id == "s1"

    def test_a_larger_k_flattens_the_top_rank(self) -> None:
        a = reciprocal_rank_fusion([[_hit("x")]], k=0)[0].score
        b = reciprocal_rank_fusion([[_hit("x")]], k=1000)[0].score
        assert a == pytest.approx(1.0) and b == pytest.approx(1 / 1001)

    def test_metadata_comes_from_the_first_ranking_containing_the_id(self) -> None:
        fused = reciprocal_rank_fusion(
            [[_hit("a", src="dense")], [_hit("a", src="sparse")]]
        )
        assert fused[0].metadata == {"src": "dense"}

    def test_a_repeated_id_within_one_ranking_counts_once_at_its_best_rank(
        self,
    ) -> None:
        fused = reciprocal_rank_fusion([[_hit("a"), _hit("a"), _hit("b")]], k=60)
        assert {h.id: h.score for h in fused} == pytest.approx(
            {"a": 1 / 61, "b": 1 / 63}
        )

    def test_top_k_and_empty_inputs(self) -> None:
        rankings = [[_hit("a"), _hit("b"), _hit("c")]]
        assert len(reciprocal_rank_fusion(rankings, top_k=2)) == 2
        assert (
            reciprocal_rank_fusion([]) == [] and reciprocal_rank_fusion([[], []]) == []
        )

    def test_ties_break_on_id(self) -> None:
        fused = reciprocal_rank_fusion([[_hit("b")], [_hit("a")]])
        assert [h.id for h in fused] == ["a", "b"]

    def test_inputs_are_not_modified(self) -> None:
        dense = [_hit("a", 0.5)]
        reciprocal_rank_fusion([dense, [_hit("a")]])
        assert dense[0].score == 0.5

    def test_bad_arguments_are_refused(self) -> None:
        with pytest.raises(ValueError, match="k must"):
            reciprocal_rank_fusion([], k=-1)
        with pytest.raises(ValueError, match="one weight"):
            reciprocal_rank_fusion([[_hit("a")]], weights=[1, 2])


@settings(max_examples=60, deadline=None)
@given(
    a=st.lists(st.sampled_from("abcdef"), unique=True, max_size=6),
    b=st.lists(st.sampled_from("abcdef"), unique=True, max_size=6),
)
def test_fusion_returns_exactly_the_union_and_never_loses_a_top_hit(
    a: list[str], b: list[str]
) -> None:
    fused = reciprocal_rank_fusion([[_hit(i) for i in a], [_hit(i) for i in b]])
    assert {h.id for h in fused} == set(a) | set(b)
    assert len(fused) == len(set(a) | set(b))
    scores = [h.score for h in fused]
    assert scores == sorted(scores, reverse=True)
