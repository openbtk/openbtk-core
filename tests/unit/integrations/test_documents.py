"""ClinicalTextChunk <-> langchain Document, against the real Document class."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("langchain_core")

from langchain_core.documents import Document

from openbtk.core.errors import ProcessingError
from openbtk.core.schemas import LinkedEntity, TextSpan
from openbtk.data.clinical_text.schemas import ClinicalTextChunk
from openbtk.integrations.langchain import (
    chunk_to_document,
    document_to_chunk,
)


def _chunk(**over: Any) -> ClinicalTextChunk:
    base: dict[str, Any] = {
        "chunk_id": "n1:0",
        "record_id": "n1",
        "text": "Plan: rest and fluids.",
        "span": TextSpan(start=10, end=32, label="chunk", confidence=1.0),
        "section": "Plan",
        "token_count": 6,
        "metadata": {"source": "synthea", "page": 2},
    }
    base.update(over)
    return ClinicalTextChunk(**base)


class TestChunkToDocument:
    def test_fields_land_where_langchain_expects_them(self) -> None:
        doc = chunk_to_document(_chunk())
        assert isinstance(doc, Document)
        assert doc.page_content == "Plan: rest and fluids."
        assert doc.id == "n1:0"
        assert doc.metadata["record_id"] == "n1"
        assert doc.metadata["section"] == "Plan"
        assert doc.metadata["token_count"] == 6
        assert (doc.metadata["span_start"], doc.metadata["span_end"]) == (10, 32)
        assert doc.metadata["source"] == "synthea"

    def test_a_chunk_without_section_or_entities_omits_those_keys(self) -> None:
        doc = chunk_to_document(_chunk(section=None, metadata={}))
        assert "section" not in doc.metadata
        assert "entities" not in doc.metadata

    def test_metadata_colliding_with_a_reserved_key_is_refused_not_overwritten(
        self,
    ) -> None:
        with pytest.raises(ProcessingError, match="reserved") as exc:
            chunk_to_document(_chunk(metadata={"record_id": "mine"}))
        assert exc.value.context["reserved_keys"] == ["record_id"]
        # The message names keys, never the chunk's text.
        assert "rest and fluids" not in str(exc.value)


class TestRoundTrip:
    def test_chunk_survives_a_round_trip_unchanged(self) -> None:
        original = _chunk()
        assert document_to_chunk(chunk_to_document(original)) == original

    def test_entities_survive_the_round_trip(self) -> None:
        entity = LinkedEntity(
            text="fluids",
            span=TextSpan(start=0, end=6, label="CHEMICAL", confidence=0.9),
            cui="C0016286",
            codes={"SNOMED": "1"},
        )
        original = _chunk(entities=[entity])
        assert document_to_chunk(chunk_to_document(original)).entities == [entity]


class TestDocumentToChunk:
    def test_a_foreign_document_needs_a_record_id_and_a_token_counter(self) -> None:
        doc = Document(id="d1", page_content="two words")
        chunk = document_to_chunk(doc, record_id="r9", token_counter=lambda t: 2)
        assert (chunk.chunk_id, chunk.record_id, chunk.token_count) == ("d1", "r9", 2)
        # With no offsets given the span covers the whole text.
        assert (chunk.span.start, chunk.span.end) == (0, len("two words"))

    def test_token_count_is_never_estimated(self) -> None:
        with pytest.raises(ProcessingError, match="never estimated"):
            document_to_chunk(Document(id="d1", page_content="x"), record_id="r")

    def test_missing_record_id_is_an_error(self) -> None:
        with pytest.raises(ProcessingError, match="record_id"):
            document_to_chunk(
                Document(id="d1", page_content="x"), token_counter=lambda t: 1
            )

    def test_missing_chunk_id_is_an_error(self) -> None:
        with pytest.raises(ProcessingError, match="neither an id"):
            document_to_chunk(
                Document(page_content="x"), record_id="r", token_counter=lambda t: 1
            )

    def test_an_explicit_record_id_wins_over_metadata(self) -> None:
        doc = chunk_to_document(_chunk())
        assert document_to_chunk(doc, record_id="other").record_id == "other"

    def test_an_invalid_chunk_is_refused_without_echoing_its_content(self) -> None:
        doc = Document(
            id="d1",
            page_content="SECRET",
            metadata={"span_start": -5, "token_count": 1},
        )
        with pytest.raises(ProcessingError, match="not a valid chunk") as exc:
            document_to_chunk(doc, record_id="r")
        assert "SECRET" not in str(exc.value)
