"""Unit tests for openbtk.retrieval.reranker.ConceptOverlapReranker.

No optional dependency, no mocking needed: ``extract_concepts`` is an
injected callable, so every test supplies its own small, deterministic
one -- exactly the point of that design (see the module's own docstring).
"""

from __future__ import annotations

from openbtk.core.schemas import SearchResult
from openbtk.retrieval.reranker import ConceptOverlapReranker


def _result(
    record_id: str, score: float, cuis: list[str] | None = None
) -> SearchResult:
    metadata = {"cuis": cuis} if cuis is not None else {}
    return SearchResult(id=record_id, score=score, metadata=metadata)


def _word_set_extractor(text: str) -> set[str]:
    """A trivial, deterministic stand-in for a real UMLS linker: treats
    each whitespace-separated token as its own "concept"."""
    return set(text.split())


class TestRerankOrdering:
    def test_more_shared_concepts_ranks_higher_even_with_a_lower_score(self) -> None:
        reranker = ConceptOverlapReranker(extract_concepts=_word_set_extractor)
        low_score_high_overlap = _result("a", score=0.1, cuis=["diabetes", "insulin"])
        high_score_no_overlap = _result("b", score=0.9, cuis=["fracture"])
        results = reranker.rerank(
            "diabetes insulin", [high_score_no_overlap, low_score_high_overlap]
        )
        assert [r.id for r in results] == ["a", "b"]

    def test_ties_in_overlap_break_by_original_score(self) -> None:
        reranker = ConceptOverlapReranker(extract_concepts=_word_set_extractor)
        lower = _result("low", score=0.2, cuis=["diabetes"])
        higher = _result("high", score=0.8, cuis=["diabetes"])
        results = reranker.rerank("diabetes", [lower, higher])
        assert [r.id for r in results] == ["high", "low"]

    def test_no_concept_metadata_anywhere_degrades_to_original_score_order(
        self,
    ) -> None:
        reranker = ConceptOverlapReranker(extract_concepts=_word_set_extractor)
        low = _result("low", score=0.1)
        high = _result("high", score=0.9)
        results = reranker.rerank("diabetes", [low, high])
        assert [r.id for r in results] == ["high", "low"]

    def test_more_overlapping_concepts_outranks_fewer(self) -> None:
        reranker = ConceptOverlapReranker(extract_concepts=_word_set_extractor)
        one_match = _result("one", score=0.5, cuis=["diabetes"])
        two_matches = _result("two", score=0.5, cuis=["diabetes", "insulin"])
        results = reranker.rerank("diabetes insulin", [one_match, two_matches])
        assert [r.id for r in results] == ["two", "one"]


class TestTopK:
    def test_top_k_truncates_the_result_list(self) -> None:
        reranker = ConceptOverlapReranker(extract_concepts=_word_set_extractor)
        results = reranker.rerank(
            "diabetes",
            [_result(str(i), score=float(i)) for i in range(5)],
            top_k=2,
        )
        assert len(results) == 2

    def test_no_top_k_returns_every_input_result(self) -> None:
        reranker = ConceptOverlapReranker(extract_concepts=_word_set_extractor)
        inputs = [_result(str(i), score=float(i)) for i in range(5)]
        results = reranker.rerank("diabetes", inputs)
        assert len(results) == 5


class TestEmptyInput:
    def test_empty_results_returns_empty_list(self) -> None:
        reranker = ConceptOverlapReranker(extract_concepts=_word_set_extractor)
        assert reranker.rerank("diabetes", []) == []

    def test_empty_results_never_calls_extract_concepts(self) -> None:
        calls: list[str] = []

        def _tracking_extractor(text: str) -> set[str]:
            calls.append(text)
            return set()

        reranker = ConceptOverlapReranker(extract_concepts=_tracking_extractor)
        reranker.rerank("diabetes", [])
        assert calls == []


class TestCustomMetadataKey:
    def test_a_custom_cuis_metadata_key_is_honoured(self) -> None:
        reranker = ConceptOverlapReranker(
            extract_concepts=_word_set_extractor, cuis_metadata_key="concepts"
        )
        match = SearchResult(id="a", score=0.1, metadata={"concepts": ["diabetes"]})
        no_match_field = _result("b", score=0.9, cuis=["diabetes"])  # wrong key
        results = reranker.rerank("diabetes", [no_match_field, match])
        assert [r.id for r in results] == ["a", "b"]


class TestExtractConceptsCalledOnce:
    def test_extract_concepts_is_called_exactly_once_per_rerank_call(self) -> None:
        calls = []

        def _tracking_extractor(text: str) -> set[str]:
            calls.append(text)
            return set(text.split())

        reranker = ConceptOverlapReranker(extract_concepts=_tracking_extractor)
        reranker.rerank(
            "diabetes",
            [_result(str(i), score=float(i), cuis=["diabetes"]) for i in range(5)],
        )
        assert calls == ["diabetes"]


class TestDeclaredAttributes:
    def test_registered_under_the_expected_key(self) -> None:
        assert ConceptOverlapReranker.registry_key == "reranker.general.concept_overlap"

    def test_provenance_records_the_metadata_key(self) -> None:
        reranker = ConceptOverlapReranker(
            extract_concepts=_word_set_extractor, cuis_metadata_key="concepts"
        )
        assert reranker.provenance().config == {"cuis_metadata_key": "concepts"}
