"""An OpenBTK vector store as a ``langchain_core`` ``VectorStore``."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from openbtk.core.errors import ConfigError, RetrievalError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from langchain_core.embeddings import Embeddings

    from openbtk.core.base import BaseVectorStore
    from openbtk.core.schemas import SearchResult

_DEFAULT_TEXT_KEY = "text"


class OpenBTKVectorStore(VectorStore):
    """Expose a ``BaseVectorStore`` and an ``Embeddings`` as a LangChain
    ``VectorStore``.

    OpenBTK stores hold vectors and metadata, not documents, so each text is
    kept in the metadata under ``text_key`` (``"text"`` -- the same key
    :class:`~openbtk.pipelines.rag.RAGPipeline` reads, so one index serves both
    APIs). A caller's own metadata may not use that key.

    Scores are the underlying store's own: their scale and direction (L2
    distance falls as similarity rises; inner product rises) depend on how the
    store was built, which this adapter cannot know. For that reason
    ``similarity_search_with_relevance_scores`` (a normalised 0-1 score) is
    unsupported rather than guessed at.

    Args:
        store: The OpenBTK vector store.
        embedding: Embeds documents and queries. Wrap an OpenBTK provider with
            :class:`~openbtk.integrations.langchain.OpenBTKEmbeddings`.
        text_key: Metadata key holding each document's text.

    Example:
        >>> import numpy as np
        >>> from openbtk.core.base import BaseVectorStore
        >>> from openbtk.core.schemas import SearchResult
        >>> from langchain_core.embeddings import DeterministicFakeEmbedding
        >>> class Mem(BaseVectorStore):
        ...     def __init__(self): self.rows = {}
        ...     def upsert(self, ids, vectors, metadata):
        ...         self.rows.update(zip(ids, metadata))
        ...     def query(self, vector, top_k=5, filter=None):
        ...         return [SearchResult(id=i, score=1.0, metadata=m)
        ...                 for i, m in list(self.rows.items())[:top_k]]
        ...     def delete(self, ids): pass
        >>> vs = OpenBTKVectorStore(Mem(), DeterministicFakeEmbedding(size=4))
        >>> _ = vs.add_texts(["note A"], ids=["a"])
        >>> vs.similarity_search("anything")[0].page_content
        'note A'
    """

    def __init__(
        self,
        store: BaseVectorStore,
        embedding: Embeddings,
        *,
        text_key: str = _DEFAULT_TEXT_KEY,
    ) -> None:
        self._store = store
        self._embedding = embedding
        self._text_key = text_key

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    @property
    def store(self) -> BaseVectorStore:
        """The wrapped OpenBTK store."""
        return self._store

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,  # noqa: ARG002 -- LangChain's signature; nothing to forward
    ) -> list[str]:
        text_list = list(texts)
        if not text_list:
            return []
        if metadatas is not None and len(metadatas) != len(text_list):
            raise ConfigError(
                f"metadatas has {len(metadatas)} entries for {len(text_list)} texts."
            )
        if ids is not None and len(ids) != len(text_list):
            raise ConfigError(f"ids has {len(ids)} entries for {len(text_list)} texts.")
        rows: list[dict[str, Any]] = []
        for i, text in enumerate(text_list):
            meta = dict(metadatas[i]) if metadatas is not None else {}
            if self._text_key in meta:
                raise ConfigError(
                    f"Metadata may not use the key {self._text_key!r}: it holds "
                    "the document text. Pass text_key= to use another."
                )
            meta[self._text_key] = text
            rows.append(meta)
        resolved = ids if ids is not None else [str(uuid.uuid4()) for _ in text_list]
        vectors = np.asarray(
            self._embedding.embed_documents(text_list), dtype=np.float32
        )
        self._store.upsert(resolved, vectors, rows)
        return resolved

    def delete(
        self,
        ids: list[str] | None = None,
        **kwargs: Any,  # noqa: ARG002 -- LangChain's signature
    ) -> bool | None:
        """Delete by id. Deleting everything (``ids=None``) is unsupported:
        ``BaseVectorStore`` has no clear operation."""
        if ids is None:
            raise NotImplementedError(
                "OpenBTKVectorStore.delete() needs explicit ids; "
                "BaseVectorStore has no delete-all."
            )
        self._store.delete(ids)
        return True

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k, **kwargs)]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        return self._search(self._embedding.embed_query(query), k, kwargs)

    def similarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[Document]:
        return [doc for doc, _ in self._search(embedding, k, kwargs)]

    def _search(
        self, vector: list[float], k: int, kwargs: dict[str, Any]
    ) -> list[tuple[Document, float]]:
        results = self._store.query(
            np.asarray(vector, dtype=np.float32),
            top_k=k,
            filter=kwargs.get("filter"),
        )
        return [(self._to_document(r), r.score) for r in results]

    def _to_document(self, result: SearchResult) -> Document:
        if self._text_key not in result.metadata:
            raise RetrievalError(
                f"Stored item {result.id!r} has no {self._text_key!r} metadata, so "
                "it cannot be returned as a Document. It was likely indexed "
                "outside this adapter without its text.",
                context={"id": result.id, "text_key": self._text_key},
            )
        metadata = {k: v for k, v in result.metadata.items() if k != self._text_key}
        return Document(
            id=result.id,
            page_content=str(result.metadata[self._text_key]),
            metadata=metadata,
        )

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        store: BaseVectorStore | None = None,
        **kwargs: Any,  # noqa: ARG003 -- LangChain's signature
    ) -> OpenBTKVectorStore:
        """Build over ``store`` and index ``texts``.

        ``store`` is required: this adapter does not choose a backend (FAISS,
        Chroma, Qdrant...) on the caller's behalf.
        """
        if store is None:
            raise ConfigError(
                "OpenBTKVectorStore.from_texts() needs store=<a BaseVectorStore>; "
                "it does not pick a backend for you."
            )
        instance = cls(store, embedding)
        instance.add_texts(texts, metadatas, ids=ids)
        return instance
