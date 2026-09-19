"""OpenBTKVectorStore through the real langchain-core VectorStore API,
including LangChain's own retriever wrapper."""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from openbtk.core.errors import ConfigError, RetrievalError
from openbtk.core.schemas import SearchResult, TextSpan
from openbtk.data.clinical_text.schemas import ClinicalTextChunk
from openbtk.integrations.langchain import (
    OpenBTKEmbeddings,
    OpenBTKVectorStore,
    chunk_to_document,
)

from ._doubles import BruteForceStore, OneHotEmbedding


def _vs() -> tuple[OpenBTKVectorStore, BruteForceStore]:
    store = BruteForceStore()
    return OpenBTKVectorStore(store, OpenBTKEmbeddings(OneHotEmbedding())), store


class TestAddAndSearch:
    def test_is_a_langchain_vector_store(self) -> None:
        assert isinstance(_vs()[0], VectorStore)

    def test_add_texts_returns_the_ids_and_stores_text_in_metadata(self) -> None:
        vs, store = _vs()
        ids = vs.add_texts(["apple"], [{"k": 1}], ids=["a"])
        assert ids == ["a"]
        assert store.rows["a"][1] == {"k": 1, "text": "apple"}

    def test_ids_are_generated_when_not_given(self) -> None:
        vs, store = _vs()
        ids = vs.add_texts(["apple", "banana"])
        assert len(set(ids)) == 2 and set(ids) == set(store.rows)

    def test_similarity_search_returns_the_nearest_documents(self) -> None:
        vs, _ = _vs()
        vs.add_texts(["apple", "banana", "cherry"], ids=["a", "b", "c"])
        docs = vs.similarity_search("avocado", k=2)
        assert docs[0] == Document(id="a", page_content="apple")
        assert len(docs) == 2

    def test_scores_are_the_stores_own(self) -> None:
        vs, _ = _vs()
        vs.add_texts(["apple", "banana"], ids=["a", "b"])
        (doc, score), *_ = vs.similarity_search_with_score("apricot", k=1)
        assert (doc.id, score) == ("a", 1.0)

    def test_search_by_vector(self) -> None:
        vs, _ = _vs()
        vs.add_texts(["apple", "banana"], ids=["a", "b"])
        assert vs.similarity_search_by_vector([0, 1, 0, 0], k=1)[0].id == "b"

    def test_filter_is_forwarded_to_the_store(self) -> None:
        vs, store = _vs()
        vs.add_texts(["apple", "avocado"], [{"src": "x"}, {"src": "y"}], ids=["a", "b"])
        docs = vs.similarity_search("apple", k=5, filter={"src": "y"})
        assert [d.id for d in docs] == ["b"]
        assert store.last_filter == {"src": "y"}

    def test_empty_add_is_a_no_op(self) -> None:
        vs, store = _vs()
        assert vs.add_texts([]) == []
        assert store.rows == {}

    def test_delete_by_id(self) -> None:
        vs, store = _vs()
        vs.add_texts(["apple", "banana"], ids=["a", "b"])
        assert vs.delete(["a"]) is True
        assert list(store.rows) == ["b"]

    def test_delete_all_is_unsupported_not_silently_ignored(self) -> None:
        with pytest.raises(NotImplementedError, match="explicit ids"):
            _vs()[0].delete()


class TestChunksThroughTheStore:
    def test_chunk_documents_index_and_come_back_with_their_provenance(self) -> None:
        chunk = ClinicalTextChunk(
            chunk_id="n1:0",
            record_id="n1",
            text="apple plan",
            span=TextSpan(start=0, end=10, label="chunk", confidence=1.0),
            token_count=2,
        )
        vs, store = _vs()
        vs.add_documents([chunk_to_document(chunk)])
        (hit,) = vs.similarity_search("apple", k=1)
        assert hit.id == "n1:0"
        assert hit.page_content == "apple plan"
        assert hit.metadata["record_id"] == "n1"
        # Same "text"/"record_id" keys RAGPipeline reads -> one index, two APIs.
        assert store.rows["n1:0"][1]["text"] == "apple plan"


class TestAsRetriever:
    def test_langchains_retriever_wrapper_works(self) -> None:
        vs, _ = _vs()
        vs.add_texts(["apple", "banana"], ids=["a", "b"])
        retriever = vs.as_retriever(search_kwargs={"k": 1})
        assert [d.id for d in retriever.invoke("avocado")] == ["a"]

    def test_normalised_relevance_scores_are_unsupported_not_guessed(self) -> None:
        vs, _ = _vs()
        vs.add_texts(["apple"], ids=["a"])
        with pytest.raises(NotImplementedError):
            vs.similarity_search_with_relevance_scores("apple")


class TestErrors:
    def test_metadata_may_not_use_the_text_key(self) -> None:
        with pytest.raises(ConfigError, match="text_key"):
            _vs()[0].add_texts(["x"], [{"text": "clash"}])

    def test_a_custom_text_key_frees_the_default(self) -> None:
        store = BruteForceStore()
        vs = OpenBTKVectorStore(
            store, OpenBTKEmbeddings(OneHotEmbedding()), text_key="body"
        )
        vs.add_texts(["apple"], [{"text": "fine"}], ids=["a"])
        assert store.rows["a"][1] == {"text": "fine", "body": "apple"}

    def test_length_mismatches_are_refused(self) -> None:
        vs, _ = _vs()
        with pytest.raises(ConfigError, match="metadatas"):
            vs.add_texts(["a", "b"], [{}])
        with pytest.raises(ConfigError, match="ids"):
            vs.add_texts(["a", "b"], ids=["only-one"])

    def test_an_item_indexed_without_text_cannot_become_a_document(self) -> None:
        vs, store = _vs()
        store.upsert(["z"], OneHotEmbedding().embed(["zeta"]), [{"k": 1}])
        with pytest.raises(RetrievalError, match="no 'text' metadata") as exc:
            vs.similarity_search("zeta")
        assert exc.value.context["id"] == "z"


class TestFromTexts:
    def test_builds_and_indexes_over_the_given_store(self) -> None:
        store = BruteForceStore()
        vs = OpenBTKVectorStore.from_texts(
            ["apple", "banana"],
            OpenBTKEmbeddings(OneHotEmbedding()),
            ids=["a", "b"],
            store=store,
        )
        assert isinstance(vs, OpenBTKVectorStore)
        assert set(store.rows) == {"a", "b"}
        assert vs.store is store

    def test_a_store_is_required_no_backend_is_guessed(self) -> None:
        with pytest.raises(ConfigError, match="store="):
            OpenBTKVectorStore.from_texts(["a"], OpenBTKEmbeddings(OneHotEmbedding()))


def test_search_result_type_is_what_the_store_returns() -> None:
    """Guards the double: the adapter is fed real SearchResult objects."""
    store = BruteForceStore()
    store.upsert(["a"], OneHotEmbedding().embed(["apple"]), [{"text": "apple"}])
    (result,) = store.query(OneHotEmbedding().embed(["apple"])[0])
    assert isinstance(result, SearchResult)
