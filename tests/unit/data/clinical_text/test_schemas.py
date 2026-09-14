"""Unit tests for openbtk.data.clinical_text.schemas."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from openbtk.core.schemas import TextSpan
from openbtk.data.clinical_text.schemas import ClinicalTextChunk, ClinicalTextRecord
from openbtk.deid.schemas import DeidStatus


def _record(**overrides: object) -> ClinicalTextRecord:
    defaults: dict[str, object] = {
        "record_id": "note-1",
        "source": "synthea",
        "text": "Chief Complaint: chest pain.",
    }
    defaults.update(overrides)
    return ClinicalTextRecord(**defaults)


def _chunk(**overrides: object) -> ClinicalTextChunk:
    defaults: dict[str, object] = {
        "chunk_id": "note-1:0",
        "record_id": "note-1",
        "text": "Chief Complaint: chest pain.",
        "span": TextSpan(start=0, end=29, label="chunk", confidence=1.0),
        "token_count": 7,
    }
    defaults.update(overrides)
    return ClinicalTextChunk(**defaults)


class TestClinicalTextRecord:
    def test_defaults(self) -> None:
        record = _record()
        assert record.deid_status == DeidStatus.UNKNOWN
        assert record.metadata == {}
        assert record.sections is None
        assert record.note_type is None

    def test_is_frozen(self) -> None:
        record = _record()
        with pytest.raises(ValidationError, match=r"(?i)frozen"):
            record.text = "mutated"  # type: ignore[misc]

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError, match=r"(?i)extra"):
            _record(bogus=1)

    def test_sections_map_to_spans_not_strings(self) -> None:
        record = _record(
            sections={
                "chief_complaint": TextSpan(
                    start=0, end=16, label="section", confidence=1.0
                )
            }
        )
        assert record.sections is not None
        assert record.sections["chief_complaint"].end == 16

    def test_accepts_a_timezone_aware_timestamp(self) -> None:
        aware = datetime(2024, 1, 1, tzinfo=UTC)
        record = _record(timestamp=aware)
        assert record.timestamp == aware

    def test_rejects_a_naive_timestamp(self) -> None:
        """docs/05_DATA_MODALITY_SPEC.md section 1.1: naive clinical
        timestamps cause real, subtle bugs in timeline construction --
        verified as a real runtime rejection, not just a ruff DTZ lint
        catching our own code's mistakes."""
        naive = datetime(2024, 1, 1)  # noqa: DTZ001 -- the point of this test
        with pytest.raises(ValidationError, match="timezone-aware"):
            _record(timestamp=naive)

    def test_round_trips_through_json(self) -> None:
        record = _record()
        restored = ClinicalTextRecord.model_validate_json(record.model_dump_json())
        assert restored == record


class TestClinicalTextChunk:
    def test_defaults(self) -> None:
        chunk = _chunk()
        assert chunk.entities == []
        assert chunk.metadata == {}
        assert chunk.section is None

    def test_is_frozen(self) -> None:
        chunk = _chunk()
        with pytest.raises(ValidationError, match=r"(?i)frozen"):
            chunk.text = "mutated"  # type: ignore[misc]

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError, match=r"(?i)extra"):
            _chunk(bogus=1)

    def test_token_count_must_be_at_least_one(self) -> None:
        with pytest.raises(ValidationError):
            _chunk(token_count=0)

    def test_span_is_mandatory(self) -> None:
        """docs/05_DATA_MODALITY_SPEC.md section 1.1: span is what makes a
        verifiable citation possible -- there is no default, and omitting
        it must fail loudly."""
        with pytest.raises(ValidationError):
            ClinicalTextChunk(chunk_id="c1", record_id="r1", text="x", token_count=1)  # type: ignore[call-arg]

    def test_round_trips_through_json(self) -> None:
        chunk = _chunk()
        restored = ClinicalTextChunk.model_validate_json(chunk.model_dump_json())
        assert restored == chunk
