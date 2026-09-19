"""LangChain interoperability adapter -- OPTIONAL, installed via
``pip install "openbtk[langchain]"``.

This is the ONLY package in OpenBTK permitted to import LangChain, enforced by
an import-linter contract (ADR-0001). Importing it without ``langchain-core``
installed raises ``MissingDependencyError`` naming the extra; nothing else in
OpenBTK is affected, and the rest of the suite runs without it.

* ``OpenBTKEmbeddings`` -- an embedding provider as ``Embeddings``
* ``OpenBTKChatModel`` -- an LLM provider as ``BaseChatModel``
* ``OpenBTKVectorStore`` -- a vector store as ``VectorStore``
* ``chunk_to_document`` / ``document_to_chunk`` -- chunks <-> ``Document``
* ``as_runnable`` / ``from_runnable`` -- components <-> ``Runnable``
* ``as_langgraph_node`` -- a component as a LangGraph node
"""

from __future__ import annotations

from openbtk.core._lazy import require

require("langchain_core", extra="langchain")

from openbtk.integrations.langchain.chat_models import OpenBTKChatModel  # noqa: E402
from openbtk.integrations.langchain.documents import (  # noqa: E402
    chunk_to_document,
    document_to_chunk,
)
from openbtk.integrations.langchain.embeddings import OpenBTKEmbeddings  # noqa: E402
from openbtk.integrations.langchain.graph import as_langgraph_node  # noqa: E402
from openbtk.integrations.langchain.runnables import (  # noqa: E402
    RunnablePreprocessor,
    as_runnable,
    from_runnable,
)
from openbtk.integrations.langchain.vectorstores import OpenBTKVectorStore  # noqa: E402

__all__ = [
    "OpenBTKChatModel",
    "OpenBTKEmbeddings",
    "OpenBTKVectorStore",
    "RunnablePreprocessor",
    "as_langgraph_node",
    "as_runnable",
    "chunk_to_document",
    "document_to_chunk",
    "from_runnable",
]
