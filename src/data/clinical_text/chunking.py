"""
Clinical text chunking strategies.

- SectionAwareChunker: respects section boundaries, falls back to token limits
- FixedTokenChunker: simple fixed-size sliding window
- SemanticChunker: embedding-similarity based boundaries
"""
from __future__ import annotations

import uuid
from typing import Any

import structlog

from opentbtk.core.base import BaseChunker
from opentbtk.core.errors import ProcessingError
from opentbtk.core.registry import CHUNKER_REGISTRY
from .schemas import ClinicalTextRecord, ClinicalTextChunk

log = structlog.get_logger(__name__)


def _count_tokens(text: str, tokenizer: Any = None) -> int:
    """Count tokens in text.

    Uses a real BiomedicalTokenizer's subword count if provided (accurate,
    matches what the embedding/LLM model actually sees). Falls back to a
    whitespace-word approximation when no tokenizer is supplied — this keeps
    chunking usable without pulling in `transformers` for lightweight installs.

    Args:
        text: Input text.
        tokenizer: Optional BiomedicalTokenizer instance for exact counts.

    Returns:
        Token count (exact if tokenizer given, else whitespace-word approximation).
    """
    if tokenizer is not None:
        return tokenizer.count_tokens(text)
    return len(text.split())


def _split_to_token_limit(
    text: str,
    max_tokens: int,
    overlap_tokens: int = 0,
    tokenizer: Any = None,
) -> list[str]:
    """Split text into chunks of at most max_tokens words (whitespace-based).

    Splitting itself remains whitespace-based for simplicity and speed even
    when a real tokenizer is supplied for counting — subword-exact splitting
    would require re-tokenizing per candidate boundary, which is unnecessary
    precision for chunk boundaries (token *counts* should still be accurate
    via `tokenizer`, but the split points are word-granular).
    """
    words = text.split()
    chunks: list[str] = []
    step = max(1, max_tokens - overlap_tokens)
    for i in range(0, len(words), step):
        chunk_words = words[i : i + max_tokens]
        chunks.append(" ".join(chunk_words))
        if i + max_tokens >= len(words):
            break
    return chunks


@CHUNKER_REGISTRY.register("chunker.clinical_text.section_aware")
class SectionAwareChunker(BaseChunker[ClinicalTextRecord, ClinicalTextChunk]):
    """Chunk clinical notes respecting section boundaries.

    If `sections` is populated on the record (by SectionSegmenter), each
    section is chunked independently so chunks never span section boundaries.
    Falls back to FixedTokenChunker behavior on records without section info.

    Args:
        max_tokens: Maximum tokens per chunk (approximate word-count, or
            exact subword count if `tokenizer` is supplied).
        overlap_tokens: Token overlap between consecutive chunks within a
            section (default: 0, no overlap).
        min_chunk_tokens: Minimum tokens for a chunk to be yielded. Sections
            shorter than this are kept as a single chunk regardless.
        tokenizer_preset: Optional BiomedicalTokenizer preset name (e.g.
            "pubmedbert") for exact subword token counts matching the
            downstream embedding/LLM model. If None, uses a fast whitespace
            approximation (no `transformers` dependency required).
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 0,
        min_chunk_tokens: int = 10,
        tokenizer_preset: str | None = None,
    ) -> None:
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens
        self._min_chunk_tokens = min_chunk_tokens
        self._tokenizer: Any = None
        if tokenizer_preset is not None:
            from .tokenization import BiomedicalTokenizer
            self._tokenizer = BiomedicalTokenizer(preset=tokenizer_preset)

    def chunk(self, record: ClinicalTextRecord) -> list[ClinicalTextChunk]:
        sections = record.sections or {"Full Note": record.raw_text}
        chunks: list[ClinicalTextChunk] = []
        chunk_index = 0
        char_offset = 0

        for section_title, section_text in sections.items():
            if _count_tokens(section_text, self._tokenizer) <= self._max_tokens:
                # Section fits in one chunk
                sub_chunks = [section_text]
            else:
                sub_chunks = _split_to_token_limit(
                    section_text, self._max_tokens, self._overlap_tokens, self._tokenizer
                )

            for sub_text in sub_chunks:
                sub_text = sub_text.strip()
                if not sub_text or _count_tokens(sub_text, self._tokenizer) < self._min_chunk_tokens:
                    continue
                # Approximate character offsets
                char_start = record.raw_text.find(sub_text[:50], char_offset)
                if char_start == -1:
                    char_start = char_offset
                char_end = char_start + len(sub_text)

                chunks.append(ClinicalTextChunk(
                    chunk_id=str(uuid.uuid4()),
                    record_id=record.record_id,
                    text=sub_text,
                    section=section_title,
                    chunk_index=chunk_index,
                    token_count=_count_tokens(sub_text, self._tokenizer),
                    char_start=char_start,
                    char_end=char_end,
                ))
                chunk_index += 1
                char_offset = char_end

        log.debug(
            "chunker.complete",
            modality="clinical_text",
            chunker="section_aware",
            record_id=record.record_id,
            n_chunks=len(chunks),
        )
        return chunks


@CHUNKER_REGISTRY.register("chunker.clinical_text.fixed_token")
class FixedTokenChunker(BaseChunker[ClinicalTextRecord, ClinicalTextChunk]):
    """Simple fixed-size sliding window chunker.

    Ignores section structure — use SectionAwareChunker for clinical notes.
    Useful for unstructured text or when section detection is unavailable.

    Args:
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Token overlap between consecutive chunks.
        min_chunk_tokens: Minimum chunk size to emit.
        tokenizer_preset: Optional BiomedicalTokenizer preset for exact
            subword counts (e.g. "pubmedbert"). Defaults to whitespace
            approximation if None.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        min_chunk_tokens: int = 10,
        tokenizer_preset: str | None = None,
    ) -> None:
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens
        self._min_chunk_tokens = min_chunk_tokens
        self._tokenizer: Any = None
        if tokenizer_preset is not None:
            from .tokenization import BiomedicalTokenizer
            self._tokenizer = BiomedicalTokenizer(preset=tokenizer_preset)

    def chunk(self, record: ClinicalTextRecord) -> list[ClinicalTextChunk]:
        sub_texts = _split_to_token_limit(
            record.raw_text, self._max_tokens, self._overlap_tokens, self._tokenizer
        )
        chunks: list[ClinicalTextChunk] = []
        char_offset = 0

        for i, text in enumerate(sub_texts):
            text = text.strip()
            if not text or _count_tokens(text, self._tokenizer) < self._min_chunk_tokens:
                continue
            char_start = record.raw_text.find(text[:50], char_offset)
            if char_start == -1:
                char_start = char_offset
            char_end = char_start + len(text)
            chunks.append(ClinicalTextChunk(
                chunk_id=str(uuid.uuid4()),
                record_id=record.record_id,
                text=text,
                section=None,
                chunk_index=i,
                token_count=_count_tokens(text, self._tokenizer),
                char_start=char_start,
                char_end=char_end,
            ))
            char_offset = char_end

        log.debug(
            "chunker.complete",
            modality="clinical_text",
            chunker="fixed_token",
            record_id=record.record_id,
            n_chunks=len(chunks),
        )
        return chunks


@CHUNKER_REGISTRY.register("chunker.clinical_text.semantic")
class SemanticChunker(BaseChunker[ClinicalTextRecord, ClinicalTextChunk]):
    """Embedding-similarity-based semantic chunking.

    Splits text at points of low semantic similarity between consecutive
    sentences, producing chunks that are semantically coherent. Requires an
    embedding provider to compute sentence-level similarities.

    Uses scispacy sentence segmentation for biomedical-aware sentence splitting.

    Args:
        embedding_key: Registry key for the embedding provider to use for
            similarity computation (default: "embedding.clinical_text.pubmedbert").
        breakpoint_threshold: Cosine similarity below which a sentence boundary
            becomes a chunk boundary (default: 0.8).
        max_tokens: Hard upper bound on chunk size (tokens).
    """

    def __init__(
        self,
        embedding_key: str = "embedding.clinical_text.pubmedbert",
        breakpoint_threshold: float = 0.8,
        max_tokens: int = 512,
    ) -> None:
        self._embedding_key = embedding_key
        self._threshold = breakpoint_threshold
        self._max_tokens = max_tokens
        self._embedder: Any = None

    def _ensure_embedder(self) -> None:
        if self._embedder is not None:
            return
        from opentbtk.core.registry import EMBEDDING_REGISTRY
        self._embedder = EMBEDDING_REGISTRY.create(self._embedding_key)

    def chunk(self, record: ClinicalTextRecord) -> list[ClinicalTextChunk]:
        import numpy as np

        self._ensure_embedder()
        # Sentence splitting: try scispacy, fallback to simple period split
        sentences = self._split_sentences(record.raw_text)
        if len(sentences) <= 1:
            return FixedTokenChunker(max_tokens=self._max_tokens).chunk(record)

        try:
            embeddings = self._embedder.embed(sentences)  # (n_sents, dim)
        except Exception as e:
            raise ProcessingError(
                "SemanticChunker: embedding failed",
                context={"record_id": record.record_id},
            ) from e

        # Compute cosine similarities between consecutive sentences
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / (norms + 1e-9)
        similarities = (normed[:-1] * normed[1:]).sum(axis=1)

        # Group sentences into chunks at low-similarity boundaries
        chunk_sentences: list[list[str]] = []
        current: list[str] = [sentences[0]]
        for i, sim in enumerate(similarities):
            next_sent = sentences[i + 1]
            current_tokens = sum(_count_tokens(s) for s in current)
            next_tokens = _count_tokens(next_sent)
            # Split if: similarity below threshold OR chunk would exceed max_tokens
            if sim < self._threshold or (current_tokens + next_tokens > self._max_tokens):
                chunk_sentences.append(current)
                current = [next_sent]
            else:
                current.append(next_sent)
        if current:
            chunk_sentences.append(current)

        chunks: list[ClinicalTextChunk] = []
        char_offset = 0
        for i, sent_group in enumerate(chunk_sentences):
            text = " ".join(sent_group).strip()
            if not text:
                continue
            char_start = record.raw_text.find(text[:50], char_offset)
            if char_start == -1:
                char_start = char_offset
            char_end = char_start + len(text)
            chunks.append(ClinicalTextChunk(
                chunk_id=str(uuid.uuid4()),
                record_id=record.record_id,
                text=text,
                section=None,
                chunk_index=i,
                token_count=_count_tokens(text),
                char_start=char_start,
                char_end=char_end,
            ))
            char_offset = char_end

        log.debug(
            "chunker.complete",
            modality="clinical_text",
            chunker="semantic",
            record_id=record.record_id,
            n_chunks=len(chunks),
        )
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        try:
            import spacy
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            doc = nlp(text)
            return [s.text.strip() for s in doc.sents if s.text.strip()]
        except ImportError:
            import re
            return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
