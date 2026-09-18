"""Vector stores and rerankers, including concept-aware reranking.

Same registration-side-effect pattern as ``openbtk.llms``/``openbtk.
embeddings``'s own ``__init__.py`` files: importing this package is what
makes ``VECTORSTORE_REGISTRY.create("vectorstore.general.faiss")`` (etc.)
work without every caller separately importing the submodule. None of
the three imports pulls in a heavy optional dependency by itself -- the
real client/index import is lazy, inside a method, via
``openbtk.core._lazy.require``.
"""

from __future__ import annotations

from openbtk.retrieval import chroma, faiss, qdrant

__all__ = ["chroma", "faiss", "qdrant"]
