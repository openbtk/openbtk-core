# Retrieval

Vector stores need the `retrieval` extra. Stores hold vectors and metadata, not
documents: keep each chunk's text in the metadata under `"text"`.

## Vector stores

::: openbtk.retrieval.faiss.FAISSVectorStore

::: openbtk.retrieval.chroma.ChromaVectorStore

::: openbtk.retrieval.qdrant.QdrantVectorStore

## Reranking

::: openbtk.retrieval.reranker.ConceptOverlapReranker

## RAG

::: openbtk.pipelines.rag.RAGPipeline
