"""Unit tests for opentbtk.data.clinical_text.preprocessing."""
from __future__ import annotations

import uuid

import pytest

from opentbtk.data.clinical_text.schemas import ClinicalTextRecord
from opentbtk.data.clinical_text.preprocessing import (
    SectionSegmenter,
    AbbreviationExpander,
    DeidentifyProcessor,
)
from opentbtk.core.errors import ProcessingError


def _make_record(text: str) -> ClinicalTextRecord:
    return ClinicalTextRecord(
        record_id=str(uuid.uuid4()),
        source="test",
        raw_text=text,
    )


class TestSectionSegmenter:
    def test_regex_fallback_detects_standard_sections(self) -> None:
        text = (
            "Chief Complaint\nChest pain.\n\n"
            "History of Present Illness\nPatient reports onset 2 days ago.\n\n"
            "Assessment\nLikely cardiac etiology.\n"
        )
        segmenter = SectionSegmenter(use_medspacy=False)
        record = segmenter.process(_make_record(text))

        assert record.sections is not None
        assert "Chief Complaint" in record.sections
        assert "History of Present Illness" in record.sections
        assert "Assessment" in record.sections
        assert "Chest pain." in record.sections["Chief Complaint"]

    def test_no_recognized_headers_falls_back_to_full_note(self) -> None:
        text = "Just some free text without any standard headers."
        segmenter = SectionSegmenter(use_medspacy=False)
        record = segmenter.process(_make_record(text))

        assert record.sections == {"Full Note": text}

    def test_does_not_mutate_original_record(self) -> None:
        text = "Chief Complaint\nPain.\n"
        original = _make_record(text)
        segmenter = SectionSegmenter(use_medspacy=False)
        result = segmenter.process(original)

        assert original.sections is None  # frozen, unmodified
        assert result.sections is not None


class TestAbbreviationExpander:
    def test_expands_known_abbreviations(self) -> None:
        text = "Patient has hx of HTN and DM, presents with SOB."
        expander = AbbreviationExpander()
        record = expander.process(_make_record(text))

        assert "hypertension" in record.raw_text.lower()
        assert "diabetes mellitus" in record.raw_text.lower()
        assert "shortness of breath" in record.raw_text.lower()

    def test_does_not_expand_substrings_within_words(self) -> None:
        # "DM" should not match inside "ADMIN" or similar
        text = "ADMIN note: patient stable."
        expander = AbbreviationExpander()
        record = expander.process(_make_record(text))
        assert "ADMIN" in record.raw_text  # untouched

    def test_case_insensitive_expansion(self) -> None:
        text = "htn noted on exam."
        expander = AbbreviationExpander()
        record = expander.process(_make_record(text))
        assert "hypertension" in record.raw_text.lower()


class TestDeidentifyProcessor:
    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ProcessingError, match="invalid mode"):
            DeidentifyProcessor(mode="not_a_real_mode")

    def test_valid_modes_accepted(self) -> None:
        for mode in ("redact", "surrogate", "hash"):
            DeidentifyProcessor(mode=mode)  # should not raise

    def test_process_without_presidio_raises_clear_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If presidio isn't installed, should raise ProcessingError with install hint."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name.startswith("presidio"):
                raise ImportError("mocked missing presidio")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        processor = DeidentifyProcessor()
        with pytest.raises(ProcessingError, match="presidio"):
            processor.process(_make_record("John Smith was admitted on 1/1/2023."))
