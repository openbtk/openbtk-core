"""Unit tests for opentbtk.data.clinical_text.chunking."""
from __future__ import annotations

import uuid

import pytest

from opentbtk.data.clinical_text.schemas import ClinicalTextRecord
from opentbtk.data.clinical_text.chunking import (
    SectionAwareChunker,
    FixedTokenChunker,
)


def _make_record(text: str, sections: dict[str, str] | None = None) -> ClinicalTextRecord:
    return ClinicalTextRecord(
        record_id=str(uuid.uuid4()),
        source="test",
        raw_text=text,
        sections=sections,
    )


class TestFixedTokenChunker:
    def test_chunks_respect_max_tokens(self) -> None:
        text = " ".join(f"word{i}" for i in range(1000))
        chunker = FixedTokenChunker(max_tokens=100, overlap_tokens=0, min_chunk_tokens=1)
        chunks = chunker.chunk(_make_record(text))

        assert len(chunks) > 1
        for c in chunks:
            assert c.token_count <= 100

    def test_chunk_indices_sequential(self) -> None:
        text = " ".join(f"word{i}" for i in range(300))
        chunker = FixedTokenChunker(max_tokens=50, overlap_tokens=0, min_chunk_tokens=1)
        chunks = chunker.chunk(_make_record(text))

        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_short_text_yields_single_chunk(self) -> None:
        text = "Short note text."
        chunker = FixedTokenChunker(max_tokens=512, min_chunk_tokens=1)
        chunks = chunker.chunk(_make_record(text))
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_overlap_creates_more_chunks_than_no_overlap(self) -> None:
        text = " ".join(f"word{i}" for i in range(200))
        no_overlap = FixedTokenChunker(max_tokens=50, overlap_tokens=0, min_chunk_tokens=1)
        with_overlap = FixedTokenChunker(max_tokens=50, overlap_tokens=25, min_chunk_tokens=1)

        chunks_a = no_overlap.chunk(_make_record(text))
        chunks_b = with_overlap.chunk(_make_record(text))
        assert len(chunks_b) >= len(chunks_a)

    def test_below_min_chunk_tokens_excluded(self) -> None:
        text = "one two three"
        chunker = FixedTokenChunker(max_tokens=512, min_chunk_tokens=10)
        chunks = chunker.chunk(_make_record(text))
        assert len(chunks) == 0


class TestSectionAwareChunker:
    def test_chunks_never_span_sections(self) -> None:
        sections = {
            "History": "Patient has a long history. " * 50,
            "Plan": "Continue monitoring. " * 50,
        }
        full_text = "\n\n".join(sections.values())
        record = _make_record(full_text, sections=sections)

        chunker = SectionAwareChunker(max_tokens=30, min_chunk_tokens=1)
        chunks = chunker.chunk(record)

        section_names = {c.section for c in chunks}
        assert section_names <= {"History", "Plan"}
        for c in chunks:
            assert c.section in ("History", "Plan")

    def test_small_section_kept_as_single_chunk(self) -> None:
        sections = {"Chief Complaint": "Chest pain."}
        record = _make_record("Chief Complaint\nChest pain.", sections=sections)

        chunker = SectionAwareChunker(max_tokens=512, min_chunk_tokens=1)
        chunks = chunker.chunk(record)

        assert len(chunks) == 1
        assert chunks[0].section == "Chief Complaint"
        assert chunks[0].text == "Chest pain."

    def test_no_sections_falls_back_to_full_note(self) -> None:
        text = "Unstructured note text without sections."
        record = _make_record(text, sections=None)

        chunker = SectionAwareChunker(max_tokens=512, min_chunk_tokens=1)
        chunks = chunker.chunk(record)

        assert len(chunks) == 1
        assert chunks[0].section == "Full Note"

    def test_all_chunks_respect_max_tokens(self) -> None:
        sections = {"History": " ".join(f"w{i}" for i in range(500))}
        record = _make_record(sections["History"], sections=sections)

        chunker = SectionAwareChunker(max_tokens=64, min_chunk_tokens=1)
        chunks = chunker.chunk(record)

        assert all(c.token_count <= 64 for c in chunks)

    def test_chunk_ids_are_unique(self) -> None:
        sections = {
            "A": " ".join(f"w{i}" for i in range(100)),
            "B": " ".join(f"w{i}" for i in range(100)),
        }
        record = _make_record("\n".join(sections.values()), sections=sections)
        chunker = SectionAwareChunker(max_tokens=20, min_chunk_tokens=1)
        chunks = chunker.chunk(record)

        ids = {c.chunk_id for c in chunks}
        assert len(ids) == len(chunks)
