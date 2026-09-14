"""Unit tests for openbtk.data.clinical_text.preprocessing.SectionSegmenter.

The "rule" backend is tested exhaustively (zero dependencies, always
available). The "medspacy" backend's real-model tests are
``@pytest.mark.slow`` -- see tests/conftest.py's OPENBTK_SLOW_TESTS gate.
"""

from __future__ import annotations

import pytest

from openbtk.core.errors import ProcessingError
from openbtk.data.clinical_text import preprocessing
from openbtk.data.clinical_text.preprocessing import SectionSegmenter
from openbtk.data.clinical_text.schemas import ClinicalTextRecord


def _record(text: str) -> ClinicalTextRecord:
    return ClinicalTextRecord(record_id="n1", source="synthea", text=text)


class TestRuleBackend:
    def test_single_section(self) -> None:
        record = _record("Chief Complaint:\nchest pain")
        result = SectionSegmenter().process(record)
        assert result.sections is not None
        span = result.sections["chief_complaint"]
        assert record.text[span.start : span.end] == "chest pain"

    def test_multiple_sections_do_not_swallow_each_other(self) -> None:
        text = "Chief Complaint:\nchest pain\nPlan:\nadmit for observation"
        record = _record(text)
        result = SectionSegmenter().process(record)
        assert result.sections is not None
        cc = result.sections["chief_complaint"]
        plan = result.sections["plan"]
        assert text[cc.start : cc.end] == "chest pain\n"
        assert text[plan.start : plan.end] == "admit for observation"

    def test_last_section_extends_to_end_of_text(self) -> None:
        record = _record("Plan:\nadmit")
        result = SectionSegmenter().process(record)
        assert result.sections is not None
        span = result.sections["plan"]
        assert span.end == len(record.text)

    def test_unknown_header_like_line_is_not_treated_as_a_section(self) -> None:
        """ "Please call your doctor if:" ends in a colon but isn't in the
        known vocabulary -- must not be treated as a section header."""
        record = _record("Please call your doctor if:\nsymptoms worsen")
        result = SectionSegmenter().process(record)
        assert result.sections == {}

    def test_no_headers_at_all_yields_empty_dict_not_none(self) -> None:
        """Empty dict means "processed, found nothing"; None would mean
        "never processed" -- a real, meaningful distinction."""
        record = _record("just plain text, no structure")
        result = SectionSegmenter().process(record)
        assert result.sections == {}

    def test_case_insensitive_header_matching(self) -> None:
        record = _record("CHIEF COMPLAINT:\nchest pain")
        result = SectionSegmenter().process(record)
        assert result.sections is not None
        assert "chief_complaint" in result.sections

    def test_confidence_is_the_documented_constant(self) -> None:
        record = _record("Plan:\nadmit")
        result = SectionSegmenter().process(record)
        assert result.sections is not None
        assert result.sections["plan"].confidence == preprocessing._RULE_CONFIDENCE

    def test_original_record_is_not_mutated(self) -> None:
        record = _record("Plan:\nadmit")
        SectionSegmenter().process(record)
        assert record.sections is None

    def test_process_is_stateless_across_calls(self) -> None:
        record = _record("Plan:\nadmit")
        segmenter = SectionSegmenter()
        assert segmenter.process(record) == segmenter.process(record)

    def test_alias_headers_map_to_the_same_label(self) -> None:
        """ "Physical Exam:" and "Physical Examination:" are documented as
        the same section label."""
        a = SectionSegmenter().process(_record("Physical Exam:\nnormal"))
        b = SectionSegmenter().process(_record("Physical Examination:\nnormal"))
        assert a.sections is not None
        assert b.sections is not None
        assert set(a.sections) == set(b.sections) == {"physical_exam"}


class TestMedspacyBackendMissingModel:
    def test_missing_model_raises_processing_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Doesn't need spaCy's model to genuinely be missing --
        monkeypatches spacy.load to fail exactly the way it does when the
        model hasn't been downloaded (OSError)."""
        preprocessing._medspacy_pipeline_cache = None

        class _StubSpacy:
            @staticmethod
            def load(name: str) -> None:
                raise OSError(f"[E050] Can't find model {name!r}")

        monkeypatch.setattr(
            preprocessing, "require", lambda module, *, extra: _StubSpacy()
        )
        with pytest.raises(ProcessingError, match="python -m spacy download"):
            SectionSegmenter(backend="medspacy").process(_record("Plan:\nadmit"))
        preprocessing._medspacy_pipeline_cache = None


@pytest.mark.slow
class TestMedspacyBackendReal:
    """Requires the real spaCy model + medspacy -- see this file's own
    module docstring for how to enable these."""

    def test_detects_a_known_section(self) -> None:
        record = _record(
            "Chief Complaint:\nchest pain.\n"
            "History of Present Illness:\nonset yesterday."
        )
        result = SectionSegmenter(backend="medspacy").process(record)
        assert result.sections is not None
        assert "chief_complaint" in result.sections

    def test_a_section_with_an_empty_body_is_omitted(self) -> None:
        """Two headers with nothing between them -- medspacy reports a
        zero-width body_span for the first, which must not become a
        degenerate zero-width TextSpan in the output."""
        record = _record(
            "Chief Complaint:History of Present Illness:\nonset yesterday."
        )
        result = SectionSegmenter(backend="medspacy").process(record)
        assert result.sections is not None
        assert "chief_complaint" not in result.sections
        assert "history_of_present_illness" in result.sections

    def test_pipeline_is_cached_across_calls(self) -> None:
        segmenter = SectionSegmenter(backend="medspacy")
        segmenter.process(_record("Plan:\nadmit"))
        cached_after_first = preprocessing._medspacy_pipeline_cache
        assert cached_after_first is not None
        segmenter.process(_record("Plan:\nadmit"))
        assert preprocessing._medspacy_pipeline_cache is cached_after_first
