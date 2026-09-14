"""Unit tests for openbtk.data.clinical_text.preprocessing: SectionSegmenter
and DeidPreprocessor.

SectionSegmenter's "rule" backend is tested exhaustively (zero
dependencies, always available). Its "medspacy" backend's real-model
tests are ``@pytest.mark.slow`` -- see tests/conftest.py's
OPENBTK_SLOW_TESTS gate. DeidPreprocessor needs no such gate: it only
ever uses RuleRecognizer by default, which has zero optional
dependencies.
"""

from __future__ import annotations

import pytest

from openbtk.core.errors import ProcessingError
from openbtk.data.clinical_text import preprocessing
from openbtk.data.clinical_text.preprocessing import DeidPreprocessor, SectionSegmenter
from openbtk.data.clinical_text.schemas import ClinicalTextRecord
from openbtk.deid import DeidMode, DeidStatus


def _record(text: str) -> ClinicalTextRecord:
    return ClinicalTextRecord(record_id="n1", source="synthea", text=text)


def _record_with_patient(
    record_id: str, text: str, patient_ref: str | None = None
) -> ClinicalTextRecord:
    return ClinicalTextRecord(
        record_id=record_id, source="synthea", text=text, patient_ref=patient_ref
    )


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


# Defined once with the suppression marker, then referenced by name
# everywhere below -- tests/security/test_fixture_hygiene.py scans by line
# text, so a line using the constant's NAME (not the literal digits) never
# re-trips the scanner, matching tests/unit/deid/test_rule_recognizer.py's
# own convention.
_SSN_VALUE = "123-45-6789"  # phi-fixture-ok: synthetic, unassigned test value
_SSN_TEXT = f"Patient SSN: {_SSN_VALUE}."


class TestDeidPreprocessor:
    def test_redact_mode_replaces_detected_phi(self) -> None:
        record = _record(_SSN_TEXT)
        result = DeidPreprocessor(mode=DeidMode.REDACT).process(record)
        assert result.text == "Patient SSN: [REDACTED]."

    def test_redact_mode_sets_deidentified_status(self) -> None:
        record = _record(_SSN_TEXT)
        result = DeidPreprocessor(mode=DeidMode.REDACT).process(record)
        assert result.deid_status is DeidStatus.DEIDENTIFIED

    def test_surrogate_mode_sets_surrogate_status(self) -> None:
        record = _record(_SSN_TEXT)
        result = DeidPreprocessor(mode=DeidMode.SURROGATE).process(record)
        assert result.deid_status is DeidStatus.SURROGATE

    def test_text_with_no_detectable_phi_is_unchanged(self) -> None:
        record = _record("Patient reports mild headache.")
        result = DeidPreprocessor(mode=DeidMode.REDACT).process(record)
        assert result.text == record.text

    def test_original_record_is_not_mutated(self) -> None:
        record = _record(_SSN_TEXT)
        DeidPreprocessor(mode=DeidMode.REDACT).process(record)
        assert record.text == _SSN_TEXT
        assert record.deid_status is DeidStatus.UNKNOWN

    def test_process_is_stateless_across_calls_for_redact(self) -> None:
        """REDACT never varies -- a real assertion, not the SURROGATE/HASH
        cases below, whose whole point is per-call state via a shared
        ConsistencyStore."""
        record = _record(_SSN_TEXT)
        pre = DeidPreprocessor(mode=DeidMode.REDACT)
        assert pre.process(record) == pre.process(record)

    def test_report_is_attached_to_metadata_not_discarded(self) -> None:
        record = _record(_SSN_TEXT)
        result = DeidPreprocessor(mode=DeidMode.REDACT).process(record)
        report = result.metadata["deid_report"]
        assert isinstance(report, dict)
        assert report["document_id"] == "n1"
        assert report["entity_counts"] == {"ssn": 1}

    def test_report_contains_no_phi_value(self) -> None:
        """Detection omits the matched text by construction (deid/schemas.py's
        own module docstring) -- the attached report must not leak the
        original value anywhere in its serialised form."""
        record = _record(_SSN_TEXT)
        result = DeidPreprocessor(mode=DeidMode.REDACT).process(record)
        assert _SSN_VALUE not in str(result.metadata["deid_report"])

    def test_surrogate_is_stable_for_the_same_patient_across_records(self) -> None:
        """Same (patient_ref, category, original) triple -> the same
        surrogate, even across two different documents -- the whole point
        of passing patient_id through at all (ConsistencyStore)."""
        pre = DeidPreprocessor(mode=DeidMode.SURROGATE)
        first = pre.process(
            _record_with_patient("n1", _SSN_TEXT, patient_ref="patient-A")
        )
        second = pre.process(
            _record_with_patient("n2", f"{_SSN_TEXT} again.", patient_ref="patient-A")
        )
        first_token = first.text.removeprefix("Patient SSN: ").removesuffix(".")
        assert first_token in second.text

    def test_missing_patient_ref_falls_back_to_record_id(self) -> None:
        """Documented, disclosed narrowing: two records with no patient_ref
        get independent surrogate numbering (per-record, not per-patient)
        rather than DeidPreprocessor silently refusing to run."""
        pre = DeidPreprocessor(mode=DeidMode.SURROGATE)
        first = pre.process(_record_with_patient("n1", _SSN_TEXT))
        second = pre.process(_record_with_patient("n2", _SSN_TEXT))
        assert first.text != second.text

    def test_recognizers_param_is_forwarded(self) -> None:
        """Passing an empty recognizer list disables detection entirely --
        proves the constructor genuinely forwards to DeidEngine rather than
        hardcoding its own recognizer set."""
        record = _record(_SSN_TEXT)
        result = DeidPreprocessor(mode=DeidMode.REDACT, recognizers=[]).process(record)
        assert result.text == record.text
