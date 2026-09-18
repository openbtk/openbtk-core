"""``RAGPipeline`` -- retrieve, optionally rerank, then generate, with
``SourceRef`` provenance carried from each retrieved chunk through to the
final answer (task 5.8, docs/03_ARCHITECTURE.md's RAG integration).

This is deliberately *not* another ``openbtk.pipelines.executor`` step
type: that streaming DAG executor has no ``embedding``/``vectorstore``/
``llm``-category dispatch yet (its own module docstring discloses this --
task 5.5's own integration test confirms it directly, against real
providers), and teaching it one is a real, separate, larger undertaking
than this task's own scope. ``RAGPipeline`` instead depends only on the
abstract base classes (``BaseEmbeddingProvider``, ``BaseVectorStore``,
``BaseLLMProvider``, ``BaseReranker``, all from ``core.base``) and wires
them together directly -- the ingest side of a real RAG system (load,
de-identify, segment, chunk) already runs through the real executor
today; this covers the query side, which does not.

**Indexing convention this class depends on:** each chunk's own text and
identifying fields must be present in the metadata dict passed to
``BaseVectorStore.upsert`` at index time -- there is nowhere else for
them to live, since a vector store here stores vectors and metadata, not
full documents. By default: ``"text"`` (the chunk's own text, needed to
build a grounded prompt), ``"record_id"`` (the source record's id), and
``"chunk_id"`` (falls back to the vector store's own result id when
absent, since callers commonly use the chunk id as the vector's id
directly). All three key names are configurable via the constructor for
an indexing pipeline that already uses different ones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openbtk.core.logging import get_logger
from openbtk.core.schemas import Message, RAGAnswer, SourceRef

if TYPE_CHECKING:
    from openbtk.core.base import (
        BaseEmbeddingProvider,
        BaseLLMProvider,
        BaseReranker,
        BaseVectorStore,
    )
    from openbtk.core.schemas import SearchResult

log = get_logger(__name__)

_DEFAULT_TEXT_KEY = "text"
_DEFAULT_RECORD_ID_KEY = "record_id"
_DEFAULT_CHUNK_ID_KEY = "chunk_id"


class RAGPipeline:
    """Answer a question by retrieving grounded context and generating
    against it, real chunk-level provenance attached throughout.

    Args:
        embedding: Embeds the question into the vector store's own space.
        vectorstore: Queried for the nearest indexed chunks.
        llm: Generates the final answer from the retrieved context.
        reranker: Optional. When given, ``candidate_pool_size`` results
            are fetched and reranked down to ``top_k``; without one,
            ``top_k`` results are used directly in the vector store's own
            order.
        top_k: Number of chunks to ground the answer in.
        candidate_pool_size: How many results to fetch before reranking.
            Defaults to ``top_k * 4`` when ``reranker`` is given (a real
            reranker needs a wider candidate pool to be worth running at
            all), or ``top_k`` itself otherwise.
        text_metadata_key: Metadata key holding each chunk's own text --
            see this module's own docstring for the indexing convention
            this whole class depends on.
        record_id_metadata_key: Metadata key holding the source record id.
        chunk_id_metadata_key: Metadata key holding the chunk id; falls
            back to the vector store's own result id when absent.

    No I/O happens in ``__init__`` (docs/09_CODING_STANDARDS.md rule 11):
    every real call happens inside :meth:`ask`.
    """

    def __init__(
        self,
        *,
        embedding: BaseEmbeddingProvider,
        vectorstore: BaseVectorStore,
        llm: BaseLLMProvider,
        reranker: BaseReranker | None = None,
        top_k: int = 5,
        candidate_pool_size: int | None = None,
        text_metadata_key: str = _DEFAULT_TEXT_KEY,
        record_id_metadata_key: str = _DEFAULT_RECORD_ID_KEY,
        chunk_id_metadata_key: str = _DEFAULT_CHUNK_ID_KEY,
    ) -> None:
        self._embedding = embedding
        self._vectorstore = vectorstore
        self._llm = llm
        self._reranker = reranker
        self._top_k = top_k
        self._candidate_pool_size = candidate_pool_size or (
            top_k * 4 if reranker is not None else top_k
        )
        self._text_key = text_metadata_key
        self._record_id_key = record_id_metadata_key
        self._chunk_id_key = chunk_id_metadata_key

    def ask(self, question: str, **generate_kwargs: Any) -> RAGAnswer:
        """Retrieve context for ``question``, then generate a grounded answer.

        Args:
            question: The user's question. Embedded, used as the vector
                store query, and (if a reranker is set) as the reranking
                query too.
            **generate_kwargs: Forwarded to the LLM provider's ``chat()``.

        Returns:
            The generated text plus the exact chunks it was grounded in,
            in the order given to the model.

        Raises:
            RetrievalError: If embedding or the vector store query fails.
            ProviderError: If the LLM call fails.
        """
        log.info(
            "rag.ask.start",
            top_k=self._top_k,
            candidate_pool_size=self._candidate_pool_size,
            reranking=self._reranker is not None,
        )
        query_vector = self._embedding.embed_one(question)
        candidates = self._vectorstore.query(
            query_vector, top_k=self._candidate_pool_size
        )
        if self._reranker is not None:
            candidates = self._reranker.rerank(question, candidates, top_k=self._top_k)
        else:
            candidates = candidates[: self._top_k]

        sources = [self._source_ref(c) for c in candidates]
        prompt = self._build_prompt(question, candidates)
        response = self._llm.chat(
            [Message(role="user", content=prompt)], **generate_kwargs
        )
        log.info("rag.ask.complete", sources_count=len(sources))
        return RAGAnswer(text=response.text, sources=sources, usage=response.usage)

    def _source_ref(self, result: SearchResult) -> SourceRef:
        if result.source is not None:
            return result.source
        record_id = result.metadata.get(self._record_id_key)
        chunk_id = result.metadata.get(self._chunk_id_key)
        return SourceRef(
            record_id=record_id if isinstance(record_id, str) else result.id,
            chunk_id=chunk_id if isinstance(chunk_id, str) else result.id,
        )

    def _build_prompt(self, question: str, candidates: list[SearchResult]) -> str:
        numbered = "\n\n".join(
            f"[{i}] {c.metadata.get(self._text_key, '')}"
            for i, c in enumerate(candidates, start=1)
        )
        return (
            "Answer the question using only the numbered sources below. "
            "Cite sources by their number in brackets, e.g. [1]. If the "
            "sources do not contain the answer, say so.\n\n"
            f"Sources:\n{numbered}\n\nQuestion: {question}"
        )
