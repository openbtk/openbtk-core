"""OpenBTKEmbeddings against the real langchain-core Embeddings contract, and
used inside a real LangChain vector store."""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from openbtk.core.errors import PolicyError
from openbtk.integrations.langchain import OpenBTKEmbeddings

from ._doubles import OneHotEmbedding


class TestEmbeddings:
    def test_is_a_langchain_embeddings(self) -> None:
        assert isinstance(OpenBTKEmbeddings(OneHotEmbedding()), Embeddings)

    def test_embed_query_returns_plain_python_floats(self) -> None:
        vec = OpenBTKEmbeddings(OneHotEmbedding()).embed_query("alpha")
        assert vec == [1.0, 0.0, 0.0, 0.0]
        assert all(type(x) is float for x in vec)

    def test_embed_documents_returns_nested_python_lists(self) -> None:
        vecs = OpenBTKEmbeddings(OneHotEmbedding()).embed_documents(["a", "b"])
        assert vecs == [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        assert type(vecs[0][0]) is float

    def test_documents_are_sent_in_slices_of_the_providers_batch_size(self) -> None:
        provider = OneHotEmbedding(batch=2)
        vecs = OpenBTKEmbeddings(provider).embed_documents(list("abcde"))
        assert provider.call_sizes == [2, 2, 1]
        assert len(vecs) == 5

    def test_empty_input_never_calls_the_provider(self) -> None:
        provider = OneHotEmbedding()
        assert OpenBTKEmbeddings(provider).embed_documents([]) == []
        assert provider.call_sizes == []

    def test_provider_is_exposed(self) -> None:
        provider = OneHotEmbedding()
        assert OpenBTKEmbeddings(provider).provider is provider

    def test_from_registry_builds_a_registered_provider_without_loading_it(
        self,
    ) -> None:
        import openbtk.embeddings  # noqa: F401 -- registers the providers

        emb = OpenBTKEmbeddings.from_registry(
            "embedding.general.huggingface",
            model="org/model",
            revision="0" * 40,
            dimension=8,
        )
        # Constructor does no I/O (CLAUDE.md rule 11): nothing was downloaded.
        assert emb.provider.registry_key == "embedding.general.huggingface"
        assert emb.provider.dimension == 8

    def test_the_offsite_guard_still_applies_through_the_adapter(self) -> None:
        import openbtk.embeddings  # noqa: F401 -- registers the providers

        with pytest.raises(PolicyError, match="sends data offsite"):
            OpenBTKEmbeddings.from_registry(
                "embedding.general.openai", api_key="x", model="m"
            )

    def test_works_inside_a_real_langchain_vector_store(self) -> None:
        """Interop, not just type-compat: LangChain's own InMemoryVectorStore
        embeds and searches through the adapter."""
        store = InMemoryVectorStore(OpenBTKEmbeddings(OneHotEmbedding()))
        store.add_texts(["apple", "banana"])
        assert store.similarity_search("avocado", k=1)[0].page_content == "apple"
