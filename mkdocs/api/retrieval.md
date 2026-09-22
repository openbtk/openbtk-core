# Retrieval

Vector stores need the `retrieval` extra. Stores hold vectors and metadata, not
documents: keep each chunk's text in the metadata under `"text"`.

## Vector stores

::: openbtk.retrieval.faiss.FAISSVectorStore

::: openbtk.retrieval.chroma.ChromaVectorStore

::: openbtk.retrieval.qdrant.QdrantVectorStore

## Reranking

::: openbtk.retrieval.reranker.ConceptOverlapReranker

::: openbtk.retrieval.cross_encoder.CrossEncoderReranker

## Hybrid (dense + BM25)

See the [retrieval guide](../guides/retrieval.md) for how the pieces fit and what BM25
will and will not find.

::: openbtk.retrieval.hybrid.BM25Index

::: openbtk.retrieval.hybrid.reciprocal_rank_fusion

::: openbtk.retrieval.hybrid.default_tokenizer

## RAG

::: openbtk.pipelines.rag.RAGPipeline
