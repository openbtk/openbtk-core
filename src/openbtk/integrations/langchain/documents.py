"""``ClinicalTextChunk`` <-> ``langchain_core.documents.Document``.

The conversion is lossless in both directions for every field a chunk has, so
a chunk can be handed to a LangChain pipeline and recovered afterwards. The
chunk's own fields live under the reserved metadata keys below; anything the
chunk already carries in ``metadata`` is preserved alongside them, and a chunk
whose ``metadata`` collides with a reserved key is *refused* rather than
silently overwritten.

``Document.page_content`` is the chunk text, exactly as-is. This adapter does
not de-identify anything: hand it chunks from a de-identified record, or the
PHI in them travels into whatever LangChain component receives the Document.

``document_to_chunk`` never invents a token count. ``ClinicalTextChunk``
requires an exact subword count, and a Document that did not come from a chunk
has none -- so the caller supplies a real ``token_counter`` or the conversion
fails. Whitespace-splitting "to get a number" is the anti-pattern CLAUDE.md
names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

from openbtk.core.errors import ProcessingError
from openbtk.core.schemas import LinkedEntity, TextSpan
from openbtk.data.clinical_text.schemas import ClinicalTextChunk

if TYPE_CHECKING:
    from collections.abc import Callable

_CHUNK_ID = "chunk_id"
_RECORD_ID = "record_id"
_SECTION = "section"
_TOKEN_COUNT = "token_count"
_SPAN_START = "span_start"
_SPAN_END = "span_end"
_ENTITIES = "entities"

_RESERVED = frozenset(
    {_CHUNK_ID, _RECORD_ID, _SECTION, _TOKEN_COUNT, _SPAN_START, _SPAN_END, _ENTITIES}
)


def chunk_to_document(chunk: ClinicalTextChunk) -> Document:
    """Convert a chunk to a LangChain ``Document``.

    Raises:
        ProcessingError: If ``chunk.metadata`` uses a key this adapter
            reserves (``chunk_id``, ``record_id``, ``section``,
            ``token_count``, ``span_start``, ``span_end``, ``entities``).

    Example:
        >>> from openbtk.core.schemas import TextSpan
        >>> chunk = ClinicalTextChunk(
        ...     chunk_id="n1:0", record_id="n1", text="Plan: rest.",
        ...     span=TextSpan(start=0, end=11, label="chunk", confidence=1.0),
        ...     token_count=4,
        ... )
        >>> doc = chunk_to_document(chunk)
        >>> doc.page_content, doc.id, doc.metadata["record_id"]
        ('Plan: rest.', 'n1:0', 'n1')
    """
    clash = sorted(_RESERVED & chunk.metadata.keys())
    if clash:
        raise ProcessingError(
            "ClinicalTextChunk.metadata uses keys reserved by the LangChain "
            f"adapter: {clash}. Rename them before converting.",
            context={"chunk_id": chunk.chunk_id, "reserved_keys": clash},
        )
    metadata: dict[str, Any] = dict(chunk.metadata)
    metadata[_CHUNK_ID] = chunk.chunk_id
    metadata[_RECORD_ID] = chunk.record_id
    metadata[_TOKEN_COUNT] = chunk.token_count
    metadata[_SPAN_START] = chunk.span.start
    metadata[_SPAN_END] = chunk.span.end
    if chunk.section is not None:
        metadata[_SECTION] = chunk.section
    if chunk.entities:
        metadata[_ENTITIES] = [e.model_dump(mode="json") for e in chunk.entities]
    return Document(id=chunk.chunk_id, page_content=chunk.text, metadata=metadata)


def document_to_chunk(
    document: Document,
    *,
    record_id: str | None = None,
    token_counter: Callable[[str], int] | None = None,
) -> ClinicalTextChunk:
    """Convert a ``Document`` back to a chunk.

    A Document produced by :func:`chunk_to_document` round-trips exactly.
    For any other Document, ``record_id`` and ``token_counter`` fill in what
    the Document cannot say; the chunk id falls back to ``Document.id``, and
    the span to the whole text.

    Args:
        document: The Document to convert.
        record_id: Overrides/fills ``metadata["record_id"]``.
        token_counter: Returns the exact subword count of a text. Required
            unless the Document carries ``metadata["token_count"]``.

    Raises:
        ProcessingError: If the chunk id, record id or token count cannot be
            determined, or the metadata is not valid for a chunk.
    """
    meta = dict(document.metadata)
    ctx: dict[str, Any] = {"document_id": document.id}

    chunk_id = meta.pop(_CHUNK_ID, None) or document.id
    resolved_record = record_id if record_id is not None else meta.pop(_RECORD_ID, None)
    meta.pop(_RECORD_ID, None)
    if not chunk_id:
        raise ProcessingError(
            "Cannot build a chunk: the Document has neither an id nor "
            "metadata['chunk_id'].",
            context=ctx,
        )
    if not resolved_record:
        raise ProcessingError(
            "Cannot build a chunk: pass record_id= or set metadata['record_id'].",
            context=ctx,
        )
    token_count = meta.pop(_TOKEN_COUNT, None)
    if token_count is None:
        if token_counter is None:
            raise ProcessingError(
                "Cannot build a chunk: the Document has no metadata['token_count'] "
                "and no token_counter was given. ClinicalTextChunk needs an exact "
                "subword count; it is never estimated.",
                context=ctx,
            )
        token_count = token_counter(document.page_content)

    text = document.page_content
    start = meta.pop(_SPAN_START, 0)
    end = meta.pop(_SPAN_END, len(text))
    section = meta.pop(_SECTION, None)
    entities = [LinkedEntity.model_validate(e) for e in meta.pop(_ENTITIES, [])]
    try:
        return ClinicalTextChunk(
            chunk_id=str(chunk_id),
            record_id=str(resolved_record),
            text=text,
            span=TextSpan(start=start, end=end, label="chunk", confidence=1.0),
            section=section,
            token_count=token_count,
            entities=entities,
            metadata=meta,
        )
    except ValueError as e:
        raise ProcessingError(
            f"Document is not a valid chunk: {type(e).__name__}.", context=ctx
        ) from e
