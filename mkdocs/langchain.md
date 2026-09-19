# LangChain and LangGraph

OpenBTK does not depend on LangChain and does not fork it
([ADR-0001](https://github.com/openbtk/openbtk-core/blob/main/docs/adr/0001-langchain-interop-not-fork.md)).
Interoperation is an **optional adapter**, `openbtk.integrations.langchain`,
which is the only package in OpenBTK allowed to import LangChain:

```bash
pip install "openbtk[langchain]"            # embeddings, chat model, vector store, runnables
pip install "openbtk[langchain,langgraph]"  # plus a real LangGraph, to run graphs
```

Without `langchain-core` installed, importing the adapter raises a
`MissingDependencyError` that names this extra, and nothing else in OpenBTK is
affected. CI checks both halves: the adapter is tested against the real
`langchain-core` and `langgraph`, and the rest of the suite is run with them
absent.

| OpenBTK | LangChain | Notes |
|---|---|---|
| `BaseEmbeddingProvider` | `Embeddings` (`OpenBTKEmbeddings`) | batches by the provider's own `batch_size` |
| `BaseLLMProvider` | `BaseChatModel` (`OpenBTKChatModel`) | system/user/assistant text only; `.stream()` yields one chunk |
| `BaseVectorStore` + embeddings | `VectorStore` (`OpenBTKVectorStore`) | scores are the store's own; no normalised relevance scores |
| `ClinicalTextChunk` | `Document` | lossless round trip (`chunk_to_document` / `document_to_chunk`) |
| components, `DeidEngine` | `Runnable` (`as_runnable`) | and a text `Runnable` back to a preprocessor (`from_runnable`) |
| components, `DeidEngine` | LangGraph node (`as_langgraph_node`) | a plain callable over the graph state |

!!! warning "De-identify before you hand data to LangChain"
    The adapter converts; it does not de-identify. A `Document` carries the
    chunk text exactly as it is. Convert chunks of an already de-identified
    record, or run `DeidEngine` as a step (below).

## Chunks and documents

```python
from openbtk.core.schemas import TextSpan
from openbtk.data.clinical_text.schemas import ClinicalTextChunk
from openbtk.integrations.langchain import chunk_to_document, document_to_chunk

chunk = ClinicalTextChunk(
    chunk_id="note-1:0",
    record_id="note-1",
    text="Plan: rest and fluids.",
    span=TextSpan(start=0, end=22, label="chunk", confidence=1.0),
    token_count=6,
)
document = chunk_to_document(chunk)
assert document.page_content == chunk.text
assert document_to_chunk(document) == chunk  # nothing is lost
```

A `Document` that did not come from a chunk has no exact token count, and
OpenBTK will not guess one: pass `token_counter=` (a real tokenizer's count)
to `document_to_chunk`, or the conversion raises.

## De-identification inside a LangChain chain

```python
from langchain_core.runnables import RunnableLambda

from openbtk.deid import DeidEngine, DeidMode
from openbtk.integrations.langchain import as_runnable

engine = DeidEngine(mode=DeidMode.REDACT, recognizers=["rule"])
deid = as_runnable(engine) | RunnableLambda(lambda result: result.text)

cleaned = deid.invoke({"text": "Call (555) 010-2345 today.", "patient_id": "hashed-id"})
assert "010-2345" not in cleaned
```

The default recognizer detects structured identifiers such as phone numbers
and dates. It does **not** detect names; add `"ner"` to `recognizers` for
that. See [Benchmarks](benchmarks.md) for what each configuration measurably
catches.

## A vector store LangChain can search

OpenBTK stores hold vectors and metadata, so the adapter keeps each text in
the metadata under `"text"` — the same key `RAGPipeline` reads. The two
classes below are tiny stand-ins so the example runs without downloading a
model; substitute a real provider and a FAISS/Chroma/Qdrant store.

```python
import numpy as np

from openbtk.core.base import BaseEmbeddingProvider, BaseVectorStore
from openbtk.core.schemas import SearchResult
from openbtk.integrations.langchain import OpenBTKEmbeddings, OpenBTKVectorStore


class LetterEmbedding(BaseEmbeddingProvider):
    """One-hot on the first letter -- a stand-in for a real model."""

    sends_data_offsite = False
    dimension = 4

    def embed(self, texts):
        out = np.zeros((len(texts), 4), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i, (ord(text[0].lower()) - ord("a")) % 4] = 1.0
        return out


class ListStore(BaseVectorStore):
    """Exact search over a dict -- a stand-in for FAISS/Chroma/Qdrant."""

    def __init__(self):
        self.rows = {}

    def upsert(self, ids, vectors, metadata):
        for i, v, m in zip(ids, vectors, metadata):
            self.rows[i] = (v, m)

    def query(self, vector, top_k=5, filter=None):
        hits = sorted(self.rows.items(), key=lambda kv: -float(kv[1][0] @ vector))
        return [
            SearchResult(id=i, score=float(v @ vector), metadata=m)
            for i, (v, m) in hits[:top_k]
        ]

    def delete(self, ids):
        for i in ids:
            self.rows.pop(i, None)


store = OpenBTKVectorStore(ListStore(), OpenBTKEmbeddings(LetterEmbedding()))
store.add_texts(["apple pie", "banana bread"], ids=["a", "b"])

retriever = store.as_retriever(search_kwargs={"k": 1})
assert retriever.invoke("avocado")[0].page_content == "apple pie"
```

## As a LangGraph node

A LangGraph node is a callable from the graph state to a partial update, so
`as_langgraph_node` returns exactly that.

```python
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from openbtk.deid import DeidEngine, DeidMode
from openbtk.integrations.langchain import as_langgraph_node


class State(TypedDict, total=False):
    request: dict[str, str]
    result: Any


graph = StateGraph(State)
graph.add_node(
    "deidentify",
    as_langgraph_node(
        DeidEngine(mode=DeidMode.REDACT, recognizers=["rule"]),
        input_key="request",
        output_key="result",
    ),
)
graph.add_edge(START, "deidentify")
graph.add_edge("deidentify", END)

state = graph.compile().invoke(
    {"request": {"text": "Call (555) 010-2345.", "patient_id": "hashed-id"}}
)
assert "010-2345" not in state["result"].text
```

## Limits worth knowing

* `OpenBTKChatModel` does not stream token-by-token: OpenBTK's provider
  `stream()` takes a prompt string, not a message list.
* `OpenBTKVectorStore.delete()` needs explicit ids (there is no delete-all on
  `BaseVectorStore`), and `from_texts()` needs an explicit `store=`.
* Wrapping a provider does not change where its data goes. An offsite provider
  built through `OpenBTKEmbeddings.from_registry` is still refused unless the
  policy allows it.
