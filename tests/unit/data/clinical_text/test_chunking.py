"""Unit and property tests for openbtk.data.clinical_text.chunking.

Each numbered item in chunking.py's own module docstring (docs/05_DATA_MODALITY_SPEC.md
section 1.3's contract) has at least one test named after it below. The two
property tests are the ones docs/07_TEST_CHARTER.md section 3.4 names
directly for chunking: "concatenated chunk texts (subset of) source text,
no character lost within a section" and "token_count <= max_tokens for
every chunk, for any input."
"""

from __future__ import annotations

import itertools

from hypothesis import given
from hypothesis import strategies as st

from openbtk.core.schemas import TextSpan
from openbtk.data.clinical_text.chunking import FixedTokenChunker, SectionAwareChunker
from openbtk.data.clinical_text.schemas import ClinicalTextRecord
from openbtk.data.clinical_text.tokenization import count_tokens_approximate


def _record(
    text: str, sections: dict[str, TextSpan] | None = None
) -> ClinicalTextRecord:
    return ClinicalTextRecord(
        record_id="n1", source="synthea", text=text, sections=sections
    )


def _span(start: int, end: int) -> TextSpan:
    return TextSpan(start=start, end=end, label="s", confidence=1.0)


class TestFixedTokenChunker:
    def test_empty_record_yields_nothing(self) -> None:
        """Degenerate case 5."""
        assert list(FixedTokenChunker().chunk(_record(""))) == []

    def test_splits_into_max_tokens_windows(self) -> None:
        record = _record("one two three four five six")
        chunks = list(FixedTokenChunker(max_tokens=2).chunk(record))
        assert [c.text.strip() for c in chunks] == [
            "one two",
            "three four",
            "five six",
        ]

    def test_token_count_matches_the_configured_counter(self) -> None:
        record = _record("one two three")
        chunks = list(FixedTokenChunker(max_tokens=2).chunk(record))
        for chunk in chunks:
            assert chunk.token_count == count_tokens_approximate(chunk.text)

    def test_no_character_is_lost(self) -> None:
        text = "one two three four five"
        chunks = list(FixedTokenChunker(max_tokens=2).chunk(_record(text)))
        assert "".join(c.text for c in chunks) == text

    def test_chunk_ids_are_sequential_per_record(self) -> None:
        record = _record("one two three four")
        chunks = list(FixedTokenChunker(max_tokens=2).chunk(record))
        assert [c.chunk_id for c in chunks] == ["n1:0", "n1:1"]

    def test_overlap_repeats_words_across_chunks(self) -> None:
        record = _record("one two three four five six")
        chunks = list(FixedTokenChunker(max_tokens=2, overlap_tokens=1).chunk(record))
        # second chunk's start should reach back into the first chunk's words
        assert "two" in chunks[1].text

    def test_a_single_very_long_word_is_kept_whole(self) -> None:
        """Disclosed edge case: a word longer than max_tokens (by whatever
        counter) is not sliced mid-word."""
        record = _record("supercalifragilisticexpialidocious")
        chunks = list(FixedTokenChunker(max_tokens=1).chunk(record))
        assert len(chunks) == 1
        assert chunks[0].text == "supercalifragilisticexpialidocious"


class TestSectionAwareChunkerDegradesToFixedToken:
    def test_no_sections_key_degrades_to_fixed_token(self) -> None:
        """Degenerate case 5: sections=None."""
        record = _record("one two three four")
        section_aware = list(SectionAwareChunker(max_tokens=2).chunk(record))
        fixed = list(FixedTokenChunker(max_tokens=2).chunk(record))
        assert [c.text for c in section_aware] == [c.text for c in fixed]

    def test_empty_sections_dict_also_degrades(self) -> None:
        record = _record("one two three four", sections={})
        section_aware = list(SectionAwareChunker(max_tokens=2).chunk(record))
        fixed = list(FixedTokenChunker(max_tokens=2).chunk(record))
        assert [c.text for c in section_aware] == [c.text for c in fixed]

    def test_empty_record_yields_nothing(self) -> None:
        assert list(SectionAwareChunker().chunk(_record(""))) == []


class TestSectionAwareChunkerRespectsSectionBoundaries:
    def test_no_chunk_spans_a_section_boundary(self) -> None:
        """Contract point 1."""
        text = "Chief Complaint:\nchest pain here.\nPlan:\nadmit patient."
        record = _record(
            text,
            sections={
                "chief_complaint": _span(17, 34),
                "plan": _span(40, 55),
            },
        )
        chunks = list(SectionAwareChunker(max_tokens=100).chunk(record))
        assert [c.section for c in chunks] == ["chief_complaint", "plan"]
        for chunk in chunks:
            if chunk.section == "chief_complaint":
                assert chunk.span.end <= 34
            else:
                assert chunk.span.start >= 40

    def test_sections_are_processed_in_document_order_regardless_of_dict_order(
        self,
    ) -> None:
        text = "Chief Complaint:\nchest pain.\nPlan:\nadmit."
        record = _record(
            text,
            sections={
                "plan": _span(35, 41),  # inserted first in the dict...
                "chief_complaint": _span(17, 29),  # ...but comes first in text
            },
        )
        chunks = list(SectionAwareChunker(max_tokens=100).chunk(record))
        assert [c.section for c in chunks] == ["chief_complaint", "plan"]


class TestSectionAwareChunkerSentenceBoundaries:
    def test_splits_at_sentence_boundaries_within_a_section(self) -> None:
        """Contract point 2."""
        text = "Findings:\nFirst sentence. Second sentence. Third sentence."
        record = _record(text, sections={"findings": _span(10, len(text))})
        chunks = list(SectionAwareChunker(max_tokens=3).chunk(record))
        assert all(chunk.text.strip().endswith(".") for chunk in chunks)

    def test_common_abbreviation_does_not_force_a_split(self) -> None:
        text = "Findings:\nSeen by Dr. Smith today for follow-up."
        record = _record(text, sections={"findings": _span(10, len(text))})
        chunks = list(SectionAwareChunker(max_tokens=100).chunk(record))
        assert len(chunks) == 1  # "Dr." did not end the sentence early

    def test_an_oversized_section_splits_internally(self) -> None:
        """Contract/degenerate case: a section longer than max_tokens
        produces more than one chunk, still bounded by section edges."""
        text = "Findings:\n" + " ".join(f"word{i}" for i in range(20))
        record = _record(text, sections={"findings": _span(10, len(text))})
        chunks = list(SectionAwareChunker(max_tokens=5).chunk(record))
        assert len(chunks) > 1
        assert all(c.section == "findings" for c in chunks)

    def test_an_oversized_single_sentence_falls_back_to_word_boundaries(self) -> None:
        """Contract point 2's explicit fallback, and degenerate case 5."""
        text = "Findings:\n" + " ".join(f"word{i}" for i in range(20)) + "."
        record = _record(text, sections={"findings": _span(10, len(text))})
        chunks = list(SectionAwareChunker(max_tokens=5).chunk(record))
        assert len(chunks) > 1
        for chunk in chunks:
            assert count_tokens_approximate(chunk.text) <= 5


class TestSectionAwareChunkerOverlap:
    def test_overlap_is_applied_within_a_section(self) -> None:
        """Contract point 4."""
        text = "Findings:\n" + " ".join(f"word{i}" for i in range(10)) + "."
        record = _record(text, sections={"findings": _span(10, len(text))})
        chunks = list(SectionAwareChunker(max_tokens=3, overlap_tokens=1).chunk(record))
        assert len(chunks) > 1
        # some word from the end of chunk i should reappear at the start of i+1
        for a, b in itertools.pairwise(chunks):
            assert a.text.split()[-1] in b.text

    def test_overlap_never_crosses_a_section_boundary(self) -> None:
        text = "Chief Complaint:\nfoo bar baz.\nPlan:\nqux quux corge."
        record = _record(
            text,
            sections={"chief_complaint": _span(17, 30), "plan": _span(36, 53)},
        )
        chunks = list(SectionAwareChunker(max_tokens=1, overlap_tokens=5).chunk(record))
        plan_chunks = [c for c in chunks if c.section == "plan"]
        assert all(c.span.start >= 36 for c in plan_chunks)


class TestTokenCountNeverZero:
    def test_whitespace_only_section_body_yields_no_chunk(self) -> None:
        """A section whose body is pure whitespace would otherwise produce
        a zero-token chunk, violating ClinicalTextChunk's ge=1 constraint
        -- must be omitted instead of silently clamped."""
        text = "Chief Complaint:   \nPlan:\nadmit."
        record = _record(
            text,
            sections={"chief_complaint": _span(17, 20), "plan": _span(26, 33)},
        )
        chunks = list(SectionAwareChunker(max_tokens=100).chunk(record))
        assert all(c.section != "chief_complaint" for c in chunks)
        assert any(c.section == "plan" for c in chunks)


class TestChunkingProperties:
    """docs/07_TEST_CHARTER.md section 3.4's named chunking properties."""

    @given(st.text(min_size=0, max_size=200), st.integers(min_value=1, max_value=20))
    def test_concatenated_chunks_reconstruct_the_source_text(
        self, text: str, max_tokens: int
    ) -> None:
        """docs/07_TEST_CHARTER.md section 3.4's literal wording is
        "concatenated chunk texts <subset of> source text" -- not exact
        equality. A text with zero real tokens (empty, or pure whitespace
        like "\\r") legitimately yields zero chunks rather than a
        fabricated one (the same, already-established rule
        TestTokenCountNeverZero exercises for a whitespace-only section
        body: ClinicalTextChunk.token_count has a real `ge=1` constraint,
        never silently clamped). Word-boundary splitting always attaches
        surrounding whitespace to a real word's chunk, so the ONLY way a
        character is dropped is when the whole text has no word at all --
        found by Hypothesis, not assumed up front."""
        chunks = list(FixedTokenChunker(max_tokens=max_tokens).chunk(_record(text)))
        joined = "".join(c.text for c in chunks)
        if text.strip():
            assert joined == text
        else:
            assert joined == ""

    @given(st.text(min_size=0, max_size=200), st.integers(min_value=1, max_value=20))
    def test_token_count_never_exceeds_max_tokens(
        self, text: str, max_tokens: int
    ) -> None:
        chunks = list(FixedTokenChunker(max_tokens=max_tokens).chunk(_record(text)))
        for chunk in chunks:
            assert chunk.token_count <= max_tokens or " " not in chunk.text.strip()
