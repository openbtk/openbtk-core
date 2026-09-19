"""``SectionAwareChunker`` and ``FixedTokenChunker``
(docs/05_DATA_MODALITY_SPEC.md section 1.3): "Generic chunkers destroy
clinical meaning. Splitting mid-way through 'Assessment and Plan' separates
a diagnosis from its treatment." This is genuinely net-new logic, not a
wrapped library -- the spec calls it "the part that matters."

**Contract, from the spec, verified one point at a time in the test
suite:**

1. If ``record.sections`` is populated, chunk each section independently
   -- no chunk ever spans a section boundary.
2. Within a section, split at sentence boundaries where possible, falling
   back to token boundaries.
3. ``token_count`` is exact when a real tokenizer is configured
   (``openbtk.data.clinical_text.tokenization.count_tokens_exact``);
   whitespace approximation is available but never the silent default of
   either chunker here (see below for why that default choice differs
   from the spec's own literal wording).
4. Configurable overlap, applied within a section only.
5. Degenerate cases: a section longer than ``max_tokens`` splits
   internally; a record with no sections degrades to fixed-token chunking
   (``FixedTokenChunker``, composed directly, not reimplemented); an empty
   record yields nothing.

**v1 defect this avoids**: its ``_split_to_token_limit()`` counted with a
real tokenizer but *split* on whitespace, so ``token_count`` and the
actual split points disagreed -- chunks could exceed ``max_tokens``
despite the field claiming otherwise. Every chunk's ``token_count`` here
is computed from the FINAL chunk text, with the SAME counter used to
decide where to cut, so the two can never disagree.

**Disclosed simplification**: "falling back to token boundaries" is
implemented at *word* granularity (whitespace-delimited), not real
subword-token offsets -- slicing mid-subword-token would produce invalid
fragments. Token *counting* is still exact when a real tokenizer is
configured; only the *cut points* are word-granular. A single word longer
than ``max_tokens`` (a very long identifier) is kept whole rather than
sliced, a real, rare edge case, not silently mishandled.

**Default token counter is the approximation, not the spec's stated
preference for exact counting.** Matches ``tokenization``'s own disclosed
deviation: nothing in this codebase defaults to needing a network
download and the ``text`` extra just by being constructed and used
(``DeidEngine``'s default recognizer set, ``NERRecognizer`` being
opt-in). Pass ``count_tokens=count_tokens_exact`` explicitly for the
spec's literal preference.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from openbtk.core.base import BaseChunker
from openbtk.core.registry import CHUNKER_REGISTRY
from openbtk.core.schemas import TextSpan
from openbtk.data.clinical_text.schemas import ClinicalTextChunk, ClinicalTextRecord
from openbtk.data.clinical_text.tokenization import count_tokens_approximate

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_ABBREVIATIONS = frozenset(
    {
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "vs",
        "etc",
        "eg",
        "ie",
        "pt",
        "pts",
        "yo",
        "y.o",
        "wk",
        "wks",
        "mo",
        "mos",
        "hr",
        "hrs",
        "min",
        "sec",
        "approx",
        "fig",
        "no",
        "vol",
        "dept",
        "univ",
        "mg",
        "ml",
        "mcg",
        "kg",
        "lb",
        "cm",
        "mm",
        "qd",
        "bid",
        "tid",
        "qid",
        "q.d",
        "b.i.d",
        "t.i.d",
        "q.i.d",
    }
)
"""A curated, disclosed-incomplete list -- real clinical abbreviation
detection is its own hard problem (medspacy/pysbd territory); this is the
rule-based approximation the module docstring already names as such."""

_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?]+(?=\s|$)")
_WORD_RE = re.compile(r"\S+\s*")


def _split_sentences(text: str) -> list[tuple[int, int]]:
    """Split ``text`` into (start, end) spans that together cover the
    WHOLE input with no gap and no overlap -- concatenating
    ``text[s:e] for s, e in spans`` always reconstructs ``text`` exactly
    (a property verified directly, not just aimed for).
    """
    if not text:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENTENCE_BOUNDARY_RE.finditer(text):
        word_before = re.search(r"(\w+)$", text[start : m.start()])
        if word_before and word_before.group(1).lower() in _ABBREVIATIONS:
            continue
        end = m.end()
        while end < len(text) and text[end].isspace():
            end += 1
        spans.append((start, end))
        start = end
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _split_by_word_boundary(
    text: str, start: int, end: int, max_tokens: int, count_tokens: Callable[[str], int]
) -> list[tuple[int, int]]:
    """Greedily pack whitespace-delimited words from ``text[start:end]``
    into chunks measuring at most ``max_tokens`` by ``count_tokens``. The
    "falling back to token boundaries" behaviour the module docstring
    describes -- see its "Disclosed simplification" note for why this cuts
    at word, not subword-token, granularity.
    """
    words = list(_WORD_RE.finditer(text, start, end))
    if not words:
        return []
    chunks: list[tuple[int, int]] = []
    piece_start = start
    cursor = start
    for w in words:
        candidate_end = w.end()
        if (
            count_tokens(text[piece_start:candidate_end]) > max_tokens
            and cursor > piece_start
        ):
            chunks.append((piece_start, cursor))
            piece_start = cursor
        cursor = candidate_end
    chunks.append((piece_start, cursor))
    return chunks


def _pack_sentences(
    text: str,
    sentences: list[tuple[int, int]],
    max_tokens: int,
    count_tokens: Callable[[str], int],
) -> list[tuple[int, int]]:
    """Greedily pack whole sentences into chunks of at most ``max_tokens``.
    A single sentence exceeding ``max_tokens`` on its own falls back to
    word-boundary splitting (spec point 2 and degenerate case 5)."""
    chunks: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end = 0
    current_tokens = 0
    for s_start, s_end in sentences:
        sentence_tokens = count_tokens(text[s_start:s_end])
        if sentence_tokens > max_tokens:
            if current_start is not None:
                chunks.append((current_start, current_end))
                current_start, current_tokens = None, 0
            chunks.extend(
                _split_by_word_boundary(text, s_start, s_end, max_tokens, count_tokens)
            )
            continue
        if current_start is None:
            current_start, current_end, current_tokens = s_start, s_end, sentence_tokens
        elif current_tokens + sentence_tokens <= max_tokens:
            current_end = s_end
            current_tokens += sentence_tokens
        else:
            chunks.append((current_start, current_end))
            current_start, current_end, current_tokens = s_start, s_end, sentence_tokens
    if current_start is not None:
        chunks.append((current_start, current_end))
    return chunks


def _apply_overlap(
    text: str,
    boundaries: list[tuple[int, int]],
    overlap_tokens: int,
    count_tokens: Callable[[str], int],
) -> list[tuple[int, int]]:
    """Extend each chunk's start backward into the previous chunk's own
    span, word by word, until adding one more word would exceed
    ``overlap_tokens`` -- applied only between chunks already known to be
    in the same section (callers only ever pass boundaries from one
    section/region at a time, so overlap never crosses a section
    boundary, per spec point 4)."""
    if overlap_tokens <= 0 or len(boundaries) < 2:
        return boundaries
    result = [boundaries[0]]
    for i in range(1, len(boundaries)):
        prev_start, _prev_end = boundaries[i - 1]
        cur_start, cur_end = boundaries[i]
        words_before = list(_WORD_RE.finditer(text, prev_start, cur_start))
        new_start = cur_start
        for w in reversed(words_before):
            candidate = text[w.start() : cur_start]
            if count_tokens(candidate) > overlap_tokens:
                break
            new_start = w.start()
        result.append((new_start, cur_end))
    return result


@CHUNKER_REGISTRY.register("chunker.clinical_text.fixed_token")
class FixedTokenChunker(BaseChunker[ClinicalTextRecord, ClinicalTextChunk]):
    """Fixed-size, token-bounded chunks -- no sentence or section
    awareness. ``SectionAwareChunker`` composes this directly for its own
    "no sections" degenerate case (spec point 5), so behaviour matches
    exactly rather than being reimplemented and risking drift.

    Example:
        >>> from openbtk.data.clinical_text.tokenization import (
        ...     count_tokens_approximate,
        ... )
        >>> record = ClinicalTextRecord(
        ...     record_id="n1", source="synthea", text="one two three four"
        ... )
        >>> chunker = FixedTokenChunker(max_tokens=2)
        >>> [c.text for c in chunker.chunk(record)] == ["one two ", "three four"]
        True
    """

    def __init__(
        self,
        *,
        max_tokens: int = 512,
        overlap_tokens: int = 0,
        count_tokens: Callable[[str], int] = count_tokens_approximate,
    ) -> None:
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens
        self._count_tokens = count_tokens

    def chunk(self, record: ClinicalTextRecord) -> Iterator[ClinicalTextChunk]:
        if not record.text:
            return
        boundaries = _split_by_word_boundary(
            record.text, 0, len(record.text), self._max_tokens, self._count_tokens
        )
        boundaries = _apply_overlap(
            record.text, boundaries, self._overlap_tokens, self._count_tokens
        )
        index = 0
        for start, end in boundaries:
            chunk = _make_chunk(record, start, end, None, index, self._count_tokens)
            if chunk is not None:
                yield chunk
                index += 1


@CHUNKER_REGISTRY.register("chunker.clinical_text.section_aware")
class SectionAwareChunker(BaseChunker[ClinicalTextRecord, ClinicalTextChunk]):
    """Section-respecting chunks -- see module docstring for the full
    contract. Degrades to ``FixedTokenChunker`` when ``record.sections``
    is empty or ``None``.

    Example:
        >>> record = ClinicalTextRecord(
        ...     record_id="n1",
        ...     source="synthea",
        ...     text="Chief Complaint:\\nchest pain.\\nPlan:\\nadmit.",
        ...     sections={
        ...         "chief_complaint": TextSpan(
        ...             start=17, end=29, label="s", confidence=1.0
        ...         ),
        ...         "plan": TextSpan(start=35, end=42, label="s", confidence=1.0),
        ...     },
        ... )
        >>> chunks = list(SectionAwareChunker(max_tokens=50).chunk(record))
        >>> [c.section for c in chunks] == ["chief_complaint", "plan"]
        True
    """

    def __init__(
        self,
        *,
        max_tokens: int = 512,
        overlap_tokens: int = 0,
        count_tokens: Callable[[str], int] = count_tokens_approximate,
    ) -> None:
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens
        self._count_tokens = count_tokens

    def chunk(self, record: ClinicalTextRecord) -> Iterator[ClinicalTextChunk]:
        if not record.text:
            return
        if not record.sections:
            fallback = FixedTokenChunker(
                max_tokens=self._max_tokens,
                overlap_tokens=self._overlap_tokens,
                count_tokens=self._count_tokens,
            )
            yield from fallback.chunk(record)
            return

        index = 0
        for label, span in sorted(
            record.sections.items(), key=lambda item: item[1].start
        ):
            sentences = _split_sentences(record.text[span.start : span.end])
            abs_sentences = [(span.start + s, span.start + e) for s, e in sentences]
            boundaries = _pack_sentences(
                record.text, abs_sentences, self._max_tokens, self._count_tokens
            )
            boundaries = _apply_overlap(
                record.text, boundaries, self._overlap_tokens, self._count_tokens
            )
            for start, end in boundaries:
                chunk = _make_chunk(
                    record, start, end, label, index, self._count_tokens
                )
                if chunk is not None:
                    yield chunk
                    index += 1


def _make_chunk(
    record: ClinicalTextRecord,
    start: int,
    end: int,
    section: str | None,
    index: int,
    count_tokens: Callable[[str], int],
) -> ClinicalTextChunk | None:
    """Returns ``None`` for a would-be zero-token chunk (e.g. a
    whitespace-only section body) rather than emitting one -- ``token_count``
    must be a real, positive, exact count
    (``ClinicalTextChunk.token_count``'s own ``ge=1`` constraint), never
    silently clamped to look valid."""
    text = record.text[start:end]
    token_count = count_tokens(text)
    if token_count < 1:
        return None
    return ClinicalTextChunk(
        chunk_id=f"{record.record_id}:{index}",
        record_id=record.record_id,
        text=text,
        span=TextSpan(start=start, end=end, label="chunk", confidence=1.0),
        section=section,
        token_count=token_count,
    )
