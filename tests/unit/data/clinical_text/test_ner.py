"""Unit tests for opentbtk.data.clinical_text.ner."""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from opentbtk.data.clinical_text.schemas import ClinicalTextChunk
from opentbtk.data.clinical_text.ner import ClinicalEntityLinker
from opentbtk.core.errors import ProcessingError


def _make_chunk(text: str) -> ClinicalTextChunk:
    return ClinicalTextChunk(
        chunk_id=str(uuid.uuid4()),
        record_id=str(uuid.uuid4()),
        text=text,
        chunk_index=0,
        token_count=len(text.split()),
        char_start=0,
        char_end=len(text),
    )


class TestClinicalEntityLinker:
    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(ProcessingError, match="Unknown scispacy model preset"):
            ClinicalEntityLinker(model_preset="not_a_real_preset")

    def test_preset_resolves_correctly(self) -> None:
        linker = ClinicalEntityLinker(model_preset="md")
        assert linker._model_name == "en_core_sci_md"

    def test_extract_with_mocked_nlp_pipeline(self) -> None:
        linker = ClinicalEntityLinker(model_preset="sm")

        mock_ent = MagicMock()
        mock_ent.text = "myocardial infarction"
        mock_ent.start_char = 10
        mock_ent.end_char = 32
        mock_ent.label_ = "DISEASE"
        mock_ent._.kb_ents = []

        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent]

        mock_nlp = MagicMock(return_value=mock_doc)
        linker._nlp = mock_nlp

        chunk = _make_chunk("Patient diagnosed with myocardial infarction today.")
        result = linker.extract(chunk)

        assert len(result.entities) == 1
        assert result.entities[0].text == "myocardial infarction"
        assert result.entities[0].label == "DISEASE"
        assert result.entities[0].cui is None  # umls linking disabled by default

    def test_extract_with_umls_linking_populates_cui(self) -> None:
        linker = ClinicalEntityLinker(model_preset="sm", link_to_umls=True)

        mock_ent = MagicMock()
        mock_ent.text = "diabetes"
        mock_ent.start_char = 0
        mock_ent.end_char = 8
        mock_ent.label_ = "DISEASE"
        mock_ent._.kb_ents = [("C0011849", 0.95)]

        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent]
        mock_nlp = MagicMock(return_value=mock_doc)
        linker._nlp = mock_nlp

        chunk = _make_chunk("diabetes mellitus type 2")
        result = linker.extract(chunk)

        assert result.entities[0].cui == "C0011849"

    def test_extract_no_entities_returns_empty_list(self) -> None:
        linker = ClinicalEntityLinker(model_preset="sm")
        mock_doc = MagicMock()
        mock_doc.ents = []
        linker._nlp = MagicMock(return_value=mock_doc)

        chunk = _make_chunk("The patient is doing well.")
        result = linker.extract(chunk)
        assert result.entities == []

    def test_pipeline_failure_raises_processing_error(self) -> None:
        linker = ClinicalEntityLinker(model_preset="sm")
        linker._nlp = MagicMock(side_effect=RuntimeError("pipeline crashed"))

        chunk = _make_chunk("Some text.")
        with pytest.raises(ProcessingError, match="Entity extraction failed"):
            linker.extract(chunk)
