"""Unit tests for openbtk.pipelines.rag.RAGPipeline.

Every dependency (embedding/vectorstore/llm/reranker) is a small,
test-local, real subclass of its actual abstract base -- not a bare
MagicMock -- so these tests exercise RAGPipeline against genuinely
interface-compliant objects, the same rationale as the M3 integration
test's own test-local guardrail doubles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from openbtk.core.base import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    BaseReranker,
    BaseVectorStore,
)
from openbtk.core.schemas import (
    LLMResponse,
    Message,
    SearchResult,
    SourceRef,
    TokenUsage,
)
from openbtk.pipelines.rag import RAGPipeline

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray


class _FakeEmbeddingProvider(BaseEmbeddingProvider):
    sends_data_offsite = False

    def __init__(self, vector: list[float] | None = None) -> None:
        self._vector = vector or [1.0, 0.0, 0.0]
        self.embedded_texts: list[str] = []

    @property
    def dimension(self) -> int:
        return len(self._vector)

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        self.embedded_texts.extend(texts)
        return np.array([self._vector for _ in texts], dtype=np.float32)


class _FakeVectorStore(BaseVectorStore):
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.last_query_top_k: int | None = None

    def upsert(
        self,
        ids: list[str],
        vectors: NDArray[np.float32],
        metadata: list[dict[str, Any]],
    ) -> None:
        raise NotImplementedError

    def query(
        self,
        vector: NDArray[np.float32],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        self.last_query_top_k = top_k
        return self._results[:top_k]

    def delete(self, ids: list[str]) -> None:
        raise NotImplementedError


class _FakeLLMProvider(BaseLLMProvider):
    sends_data_offsite = False

    def __init__(self, text: str = "an answer") -> None:
        self._text = text
        self.last_messages: list[Message] | None = None

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        return self.chat([Message(role="user", content=prompt)], **kwargs)

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        yield self._text

    def chat(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        self.last_messages = messages
        return LLMResponse(text=self._text, usage=TokenUsage(total_tokens=7))


class _FakeReranker(BaseReranker):
    def __init__(self) -> None:
        self.last_query: str | None = None
        self.last_top_k: int | None = None

    def rerank(
        self, query: str, results: list[SearchResult], top_k: int | None = None
    ) -> list[SearchResult]:
        self.last_query = query
        self.last_top_k = top_k
        reversed_results = list(reversed(results))
        return reversed_results if top_k is None else reversed_results[:top_k]


def _chunk_result(
    result_id: str, record_id: str, chunk_id: str, text: str, score: float = 1.0
) -> SearchResult:
    return SearchResult(
        id=result_id,
        score=score,
        metadata={"text": text, "record_id": record_id, "chunk_id": chunk_id},
    )


class TestAskWithoutReranking:
    def test_returns_an_answer_grounded_in_the_retrieved_sources(self) -> None:
        vectorstore = _FakeVectorStore(
            [_chunk_result("r1", "note-1", "chunk-1", "Patient has diabetes.")]
        )
        llm = _FakeLLMProvider("The patient has diabetes.")
        pipeline = RAGPipeline(
            embedding=_FakeEmbeddingProvider(),
            vectorstore=vectorstore,
            llm=llm,
        )
        answer = pipeline.ask("Does the patient have diabetes?")
        assert answer.text == "The patient has diabetes."
        assert answer.sources == [SourceRef(record_id="note-1", chunk_id="chunk-1")]

    def test_embeds_the_question_not_a_transformed_version_of_it(self) -> None:
        embedding = _FakeEmbeddingProvider()
        pipeline = RAGPipeline(
            embedding=embedding,
            vectorstore=_FakeVectorStore([]),
            llm=_FakeLLMProvider(),
        )
        pipeline.ask("What medication was prescribed?")
        assert embedding.embedded_texts == ["What medication was prescribed?"]

    def test_queries_top_k_directly_when_no_reranker_is_set(self) -> None:
        vectorstore = _FakeVectorStore(
            [_chunk_result(str(i), f"r{i}", f"c{i}", f"text {i}") for i in range(10)]
        )
        pipeline = RAGPipeline(
            embedding=_FakeEmbeddingProvider(),
            vectorstore=vectorstore,
            llm=_FakeLLMProvider(),
            top_k=3,
        )
        pipeline.ask("question")
        assert vectorstore.last_query_top_k == 3

    def test_usage_is_propagated_from_the_llm_response(self) -> None:
        pipeline = RAGPipeline(
            embedding=_FakeEmbeddingProvider(),
            vectorstore=_FakeVectorStore([_chunk_result("r1", "n1", "c1", "x")]),
            llm=_FakeLLMProvider(),
        )
        answer = pipeline.ask("question")
        assert answer.usage is not None
        assert answer.usage.total_tokens == 7


class TestPromptConstruction:
    def test_prompt_includes_numbered_sources_and_the_question(self) -> None:
        llm = _FakeLLMProvider()
        pipeline = RAGPipeline(
            embedding=_FakeEmbeddingProvider(),
            vectorstore=_FakeVectorStore(
                [
                    _chunk_result("r1", "n1", "c1", "First fact."),
                    _chunk_result("r2", "n2", "c2", "Second fact."),
                ]
            ),
            llm=llm,
        )
        pipeline.ask("What are the facts?")
        assert llm.last_messages is not None
        prompt = llm.last_messages[0].content
        assert "[1] First fact." in prompt
        assert "[2] Second fact." in prompt
        assert "What are the facts?" in prompt


class TestSourceRefReconstruction:
    def test_uses_the_configured_metadata_keys(self) -> None:
        vectorstore = _FakeVectorStore(
            [
                SearchResult(
                    id="r1",
                    score=1.0,
                    metadata={"body": "text", "doc": "note-1", "part": "chunk-1"},
                )
            ]
        )
        pipeline = RAGPipeline(
            embedding=_FakeEmbeddingProvider(),
            vectorstore=vectorstore,
            llm=_FakeLLMProvider(),
            text_metadata_key="body",
            record_id_metadata_key="doc",
            chunk_id_metadata_key="part",
        )
        answer = pipeline.ask("question")
        assert answer.sources == [SourceRef(record_id="note-1", chunk_id="chunk-1")]

    def test_falls_back_to_the_result_id_when_metadata_keys_are_missing(self) -> None:
        vectorstore = _FakeVectorStore([SearchResult(id="r1", score=1.0, metadata={})])
        pipeline = RAGPipeline(
            embedding=_FakeEmbeddingProvider(),
            vectorstore=vectorstore,
            llm=_FakeLLMProvider(),
        )
        answer = pipeline.ask("question")
        assert answer.sources == [SourceRef(record_id="r1", chunk_id="r1")]

    def test_prefers_an_already_populated_source_over_metadata(self) -> None:
        real_source = SourceRef(record_id="from-source", chunk_id="also-from-source")
        vectorstore = _FakeVectorStore(
            [
                SearchResult(
                    id="r1",
                    score=1.0,
                    metadata={"record_id": "from-metadata", "chunk_id": "ignored"},
                    source=real_source,
                )
            ]
        )
        pipeline = RAGPipeline(
            embedding=_FakeEmbeddingProvider(),
            vectorstore=vectorstore,
            llm=_FakeLLMProvider(),
        )
        answer = pipeline.ask("question")
        assert answer.sources == [real_source]


class TestReranking:
    def test_reranker_is_used_when_provided(self) -> None:
        vectorstore = _FakeVectorStore(
            [
                _chunk_result("r1", "n1", "c1", "first"),
                _chunk_result("r2", "n2", "c2", "second"),
            ]
        )
        reranker = _FakeReranker()
        pipeline = RAGPipeline(
            embedding=_FakeEmbeddingProvider(),
            vectorstore=vectorstore,
            llm=_FakeLLMProvider(),
            reranker=reranker,
            top_k=2,
        )
        answer = pipeline.ask("my question")
        assert reranker.last_query == "my question"
        assert reranker.last_top_k == 2
        # The fake reranker reverses order -- confirms its output, not the
        # vector store's own order, drives the final sources.
        assert [s.chunk_id for s in answer.sources] == ["c2", "c1"]

    def test_candidate_pool_defaults_to_four_times_top_k_when_reranking(self) -> None:
        vectorstore = _FakeVectorStore(
            [_chunk_result(str(i), f"r{i}", f"c{i}", "x") for i in range(20)]
        )
        pipeline = RAGPipeline(
            embedding=_FakeEmbeddingProvider(),
            vectorstore=vectorstore,
            llm=_FakeLLMProvider(),
            reranker=_FakeReranker(),
            top_k=3,
        )
        pipeline.ask("question")
        assert vectorstore.last_query_top_k == 12

    def test_an_explicit_candidate_pool_size_overrides_the_default(self) -> None:
        vectorstore = _FakeVectorStore(
            [_chunk_result(str(i), f"r{i}", f"c{i}", "x") for i in range(20)]
        )
        pipeline = RAGPipeline(
            embedding=_FakeEmbeddingProvider(),
            vectorstore=vectorstore,
            llm=_FakeLLMProvider(),
            reranker=_FakeReranker(),
            top_k=3,
            candidate_pool_size=5,
        )
        pipeline.ask("question")
        assert vectorstore.last_query_top_k == 5


class TestNoResults:
    def test_no_retrieved_chunks_still_produces_an_answer_with_empty_sources(
        self,
    ) -> None:
        pipeline = RAGPipeline(
            embedding=_FakeEmbeddingProvider(),
            vectorstore=_FakeVectorStore([]),
            llm=_FakeLLMProvider("I don't know."),
        )
        answer = pipeline.ask("question")
        assert answer.sources == []
        assert answer.text == "I don't know."


class TestGenerateKwargs:
    def test_extra_kwargs_are_forwarded_to_the_llm(self) -> None:
        class _RecordingLLM(_FakeLLMProvider):
            def chat(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
                self.received_kwargs = kwargs
                return super().chat(messages, **kwargs)

        llm = _RecordingLLM()
        pipeline = RAGPipeline(
            embedding=_FakeEmbeddingProvider(),
            vectorstore=_FakeVectorStore([]),
            llm=llm,
        )
        pipeline.ask("question", temperature=0.2)
        assert llm.received_kwargs == {"temperature": 0.2}


class TestHybridRetrieval:
    """``bm25=`` fuses BM25 with the vector store's ranking (FR-R-05)."""

    def _index(self) -> Any:
        from openbtk.retrieval.hybrid import BM25Index

        index = BM25Index()
        index.upsert(
            ["dense-hit", "exact-term-hit", "other"],
            [
                "The patient was seen for general follow up.",
                "Started hydroxychloroquine for rheumatoid arthritis.",
                "Discussed diet and exercise.",
            ],
            [
                {"record_id": "n1", "chunk_id": "c1"},
                {"record_id": "n2", "chunk_id": "c2"},
                {"record_id": "n3", "chunk_id": "c3"},
            ],
        )
        return index

    def _dense_only(self) -> list[SearchResult]:
        # The embedding model ranks the generic chunk first and never surfaces the
        # chunk that names the drug: the failure BM25 exists to repair.
        return [
            _chunk_result(
                "dense-hit", "n1", "c1", "The patient was seen for general follow up."
            ),
            _chunk_result("other", "n3", "c3", "Discussed diet and exercise."),
        ]

    def test_without_bm25_an_exact_term_the_embedding_missed_is_absent(self) -> None:
        pipeline = RAGPipeline(
            embedding=_FakeEmbeddingProvider(),
            vectorstore=_FakeVectorStore(self._dense_only()),
            llm=_FakeLLMProvider(),
            top_k=3,
        )
        answer = pipeline.ask("hydroxychloroquine")
        assert SourceRef(record_id="n2", chunk_id="c2") not in answer.sources

    def test_with_bm25_the_exact_term_hit_is_retrieved(self) -> None:
        pipeline = RAGPipeline(
            embedding=_FakeEmbeddingProvider(),
            vectorstore=_FakeVectorStore(self._dense_only()),
            llm=_FakeLLMProvider(),
            bm25=self._index(),
            top_k=3,
        )
        answer = pipeline.ask("hydroxychloroquine")
        assert SourceRef(record_id="n2", chunk_id="c2") in answer.sources

    def test_a_chunk_found_by_both_retrievers_ranks_first(self) -> None:
        dense = [
            _chunk_result("other", "n3", "c3", "Discussed diet and exercise."),
            _chunk_result(
                "exact-term-hit",
                "n2",
                "c2",
                "Started hydroxychloroquine for rheumatoid arthritis.",
            ),
        ]
        llm = _FakeLLMProvider()
        pipeline = RAGPipeline(
            embedding=_FakeEmbeddingProvider(),
            vectorstore=_FakeVectorStore(dense),
            llm=llm,
            bm25=self._index(),
            top_k=3,
        )
        answer = pipeline.ask("hydroxychloroquine")
        assert answer.sources[0] == SourceRef(record_id="n2", chunk_id="c2")

    def test_the_pool_is_widened_and_reranking_still_runs_after_fusion(self) -> None:
        vectorstore = _FakeVectorStore(self._dense_only())
        reranker = _FakeReranker()
        pipeline = RAGPipeline(
            embedding=_FakeEmbeddingProvider(),
            vectorstore=vectorstore,
            llm=_FakeLLMProvider(),
            reranker=reranker,
            bm25=self._index(),
            top_k=2,
        )
        pipeline.ask("hydroxychloroquine")
        assert vectorstore.last_query_top_k == 8  # top_k * 4
        assert reranker.last_top_k == 2

    def test_bm25_alone_widens_the_pool(self) -> None:
        vectorstore = _FakeVectorStore(self._dense_only())
        RAGPipeline(
            embedding=_FakeEmbeddingProvider(),
            vectorstore=vectorstore,
            llm=_FakeLLMProvider(),
            bm25=self._index(),
            top_k=2,
        ).ask("q")
        assert vectorstore.last_query_top_k == 8

    def test_an_explicit_pool_size_is_respected(self) -> None:
        vectorstore = _FakeVectorStore(self._dense_only())
        RAGPipeline(
            embedding=_FakeEmbeddingProvider(),
            vectorstore=vectorstore,
            llm=_FakeLLMProvider(),
            bm25=self._index(),
            candidate_pool_size=5,
        ).ask("q")
        assert vectorstore.last_query_top_k == 5

    def test_sources_come_from_bm25_metadata_when_the_store_did_not_return_the_chunk(
        self,
    ) -> None:
        pipeline = RAGPipeline(
            embedding=_FakeEmbeddingProvider(),
            vectorstore=_FakeVectorStore([]),
            llm=_FakeLLMProvider(),
            bm25=self._index(),
            top_k=3,
        )
        answer = pipeline.ask("hydroxychloroquine")
        assert answer.sources == [SourceRef(record_id="n2", chunk_id="c2")]
