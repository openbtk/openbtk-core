# Retrieval

`RAGPipeline` retrieves chunks for a question, optionally reranks them, and asks a model
to answer from them, carrying a `SourceRef` from each chunk to the answer. This guide is
about getting the *right chunks*, in three layers you can add one at a time.

| Layer | Finds | Costs | Extra |
|---|---|---|---|
| Vector store (dense) | chunks that *mean* something like the question | one embedding per query | `retrieval` |
| **BM25** (sparse), fused with the dense results | chunks that contain the question's *exact terms* | in-memory index | none |
| **Cross-encoder** reranker | the best few of a wider candidate list | one model pass per candidate | `text`, `llms` |

## Why a dense store alone misses things

An embedding model compresses a chunk into one vector, so a rare word (a drug name, a lab
code, an abbreviation) can get lost. BM25 ranks by the words a chunk shares with the
question, weighted by how rare they are, so it finds exactly those. Fuse the two rankings
and each covers the other's blind spot:

```python
from openbtk.core.schemas import SearchResult
from openbtk.retrieval.hybrid import BM25Index, reciprocal_rank_fusion

chunks = {
    "c1": "The patient was seen for general follow up.",
    "c2": "Started hydroxychloroquine for rheumatoid arthritis.",
    "c3": "Discussed diet and exercise.",
}
index = BM25Index()
index.upsert(
    list(chunks), list(chunks.values())
)  # index with the SAME ids as the store

# What an embedding model might return: it never surfaces c2.
dense = [SearchResult(id="c1", score=0.91), SearchResult(id="c3", score=0.74)]
sparse = index.search("hydroxychloroquine", top_k=3)
assert [hit.id for hit in sparse] == ["c2"]

fused = reciprocal_rank_fusion([dense, sparse])
assert "c2" in [hit.id for hit in fused]
```

`reciprocal_rank_fusion` combines *ranks*, not scores, because a cosine similarity and a
BM25 score are not on one scale. A chunk found by both retrievers ranks above one found by
either alone.

To use it in a pipeline, pass the index; `RAGPipeline` runs both retrievers, fuses them,
then reranks and truncates to `top_k`:

```python
import inspect

from openbtk.pipelines import RAGPipeline

# RAGPipeline(embedding=..., vectorstore=..., llm=..., bm25=index, top_k=5)
assert "bm25" in inspect.signature(RAGPipeline).parameters
```

### What to know about the BM25 index

- It is **in memory and not persisted**. Rebuild it from your chunks at start-up, in the
  same call that indexes the vector store, so the ids match.
- It holds each chunk's metadata (including its text) exactly as the vector store does, so
  it is as sensitive as your store. Deleting a chunk removes its text and terms.
- The default tokenizer lowercases and splits on anything that is not a letter or digit
  (`58410-2` becomes `58410`, `2`), with no stemming: "diabetic" does not match
  "diabetes". Pass `tokenizer=` to change it.
- It is implemented in OpenBTK rather than wrapping `rank_bm25`, whose idf is negative
  for a term in over half the documents and is patched with a floor that can itself be
  negative on a small corpus. This index uses the never-negative idf Lucene uses; the
  scoring is checked against hand-computed values.

## Reranking with a cross-encoder

A vector store scores the question and each chunk separately. A cross-encoder reads them
*together*, which is slower and better at ordering a short list. Fetch a wide pool, let it
pick the best few:

```python
from openbtk.retrieval.cross_encoder import CrossEncoderReranker

reranker = CrossEncoderReranker()  # MedCPT's cross-encoder; nothing is downloaded yet
identity = reranker.provenance().model_identity
assert identity is not None and identity.name == "ncbi/MedCPT-Cross-Encoder"
assert len(identity.revision) == 40  # pinned to a commit, never a moving branch
```

The model downloads from the Hugging Face Hub on first use, then runs **locally**: the
question and the chunk text never leave your machine. It scores the text in each
result's `metadata["text"]`; a result with none is kept after the scored ones rather than
dropped, and its original score stays in `metadata["retrieval_score"]` on scored ones.
The cross-encoder score is a raw model logit: comparable within one call, not a
probability. Give another model with `model=` and a pinned `revision=`.

Passing both to a pipeline looks like this:

```python
# RAGPipeline(..., reranker=reranker, bm25=index, top_k=5)
# fetches top_k * 4 from each retriever, fuses, reranks to top_k, then answers.
```

Whether a wider pool, BM25 or a reranker helps *your* questions is an empirical matter.
The [evaluation guide](evaluation.md) shows how to measure retrieval on your own labelled
questions before you turn any of it on.
