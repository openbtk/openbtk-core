"""
Integration test: full clinical text pipeline end-to-end.

Uses synthetic data only. Mocks heavy ML dependencies (presidio, transformers)
to keep the test fast and runnable in CI without GPU/large downloads, while
still exercising the real wiring between loaders, preprocessors, chunkers,
and guardrails.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from openbtk.data.clinical_text.loaders import PlainTextLoader
from openbtk.data.clinical_text.preprocessing import (
    AbbreviationExpander,
    SectionSegmenter,
)
from openbtk.data.clinical_text.chunking import SectionAwareChunker
from openbtk.data.clinical_text.guardrails import PHIGuardrail
from openbtk.data.clinical_text.embeddings import PubMedBERTEmbedding


@pytest.mark.integration
def test_clinical_text_pipeline_end_to_end(tmp_path: Path) -> None:
    """Load → expand abbreviations → segment sections → chunk → guardrail check."""
    note_text = (
        "Chief Complaint\n"
        "Pt c/o SOB and CP x 2 days.\n\n"
        "History of Present Illness\n"
        "Pt has hx of HTN and DM. Presents with worsening SOB.\n\n"
        "Assessment and Plan\n"
        "Likely CHF exacerbation. Start diuresis, monitor closely.\n"
    )
    note_file = tmp_path / "note.txt"
    note_file.write_text(note_text)

    # Step 1: Load
    loader = PlainTextLoader(note_type="Progress Note")
    records = loader.load_all(str(tmp_path))
    assert len(records) == 1
    record = records[0]
    assert record.note_type == "Progress Note"

    # Step 2: Expand abbreviations
    expander = AbbreviationExpander()
    record = expander.process(record)
    assert "shortness of breath" in record.raw_text.lower()
    assert "hypertension" in record.raw_text.lower()

    # Step 3: Section segmentation (regex fallback — no medspacy dependency)
    segmenter = SectionSegmenter(use_medspacy=False)
    record = segmenter.process(record)
    assert record.sections is not None
    assert "Chief Complaint" in record.sections
    assert "Assessment and Plan" in record.sections

    # Step 4: Chunking respects section boundaries
    chunker = SectionAwareChunker(max_tokens=256, min_chunk_tokens=1)
    chunks = chunker.chunk(record)
    assert len(chunks) >= 2
    assert all(c.token_count <= 256 for c in chunks)
    section_labels = {c.section for c in chunks}
    assert "Chief Complaint" in section_labels

    # Step 5: PHI guardrail check on each chunk (mocked analyzer — no PHI present)
    guard = PHIGuardrail()
    mock_analyzer = MagicMock()
    mock_analyzer.analyze.return_value = []  # synthetic note has no real PHI
    guard._analyzer = mock_analyzer

    for chunk in chunks:
        result = guard.check(chunk.text)
        assert result.passed is True


@pytest.mark.integration
def test_clinical_text_pipeline_with_embedding(tmp_path: Path) -> None:
    """Extend the pipeline with a mocked embedding step and verify output shape."""
    note_file = tmp_path / "note.txt"
    note_file.write_text(
        "Chief Complaint\nFatigue and weight loss.\n\n"
        "Assessment\nFurther workup needed to rule out malignancy.\n"
    )

    loader = PlainTextLoader()
    record = loader.load_all(str(tmp_path))[0]

    segmenter = SectionSegmenter(use_medspacy=False)
    record = segmenter.process(record)

    chunker = SectionAwareChunker(max_tokens=128, min_chunk_tokens=1)
    chunks = chunker.chunk(record)
    assert len(chunks) >= 1

    # Mock the embedding model to avoid downloading PubMedBERT in CI
    embedder = PubMedBERTEmbedding()
    embedder._dim = 768

    fake_vectors = np.random.rand(len(chunks), 768).astype(np.float32)
    embedder.embed = MagicMock(return_value=fake_vectors)  # type: ignore[method-assign]

    texts = [c.text for c in chunks]
    vectors = embedder.embed(texts)

    assert vectors.shape == (len(chunks), 768)
    assert vectors.dtype == np.float32
