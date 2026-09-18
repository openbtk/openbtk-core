"""Shared contract every registered BaseReranker must satisfy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from openbtk.core.registry import RERANKER_REGISTRY
from openbtk.core.schemas import SearchResult

if TYPE_CHECKING:
    from openbtk.core.base import BaseReranker

# ConceptOverlapReranker needs a concept-extraction callable up front (no
# default -- see its own module docstring for why); the reference
# implementation needs nothing extra.
_CONSTRUCTOR_KWARGS_BY_KEY: dict[str, dict[str, Any]] = {
    "reranker.general.concept_overlap": {
        "extract_concepts": lambda text: set(text.split())
    },
}


def _new_instance(key: str) -> BaseReranker:
    return RERANKER_REGISTRY.create(key, **_CONSTRUCTOR_KWARGS_BY_KEY.get(key, {}))


def _sample_results() -> list[SearchResult]:
    return [SearchResult(id=str(i), score=float(i)) for i in range(5)]


@pytest.mark.parametrize("key", RERANKER_REGISTRY.list_keys())
class TestRerankerContract:
    def test_rerank_returns_search_results(self, key: str) -> None:
        reranker = _new_instance(key)
        result = reranker.rerank("query", _sample_results())
        assert all(isinstance(r, SearchResult) for r in result)

    def test_rerank_never_invents_new_ids(self, key: str) -> None:
        """Reranking reorders or filters; it must never introduce a result
        id that wasn't in the input."""
        reranker = _new_instance(key)
        inputs = _sample_results()
        input_ids = {r.id for r in inputs}
        output_ids = {r.id for r in reranker.rerank("query", inputs)}
        assert output_ids <= input_ids

    def test_top_k_bounds_the_result_count(self, key: str) -> None:
        reranker = _new_instance(key)
        result = reranker.rerank("query", _sample_results(), top_k=2)
        assert len(result) <= 2

    def test_empty_input_yields_empty_output(self, key: str) -> None:
        reranker = _new_instance(key)
        assert reranker.rerank("query", []) == []

    def test_provenance_is_serialisable(self, key: str) -> None:
        reranker = _new_instance(key)
        dumped = reranker.provenance().model_dump_json()
        assert isinstance(dumped, str) and len(dumped) > 0
