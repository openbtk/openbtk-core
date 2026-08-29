"""Unit tests for opentbtk.data.clinical_text.loaders."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from opentbtk.data.clinical_text.loaders import (
    PlainTextLoader,
    MIMICNotesLoader,
    CDAClinicalLoader,
)
from opentbtk.core.errors import LoaderError


class TestPlainTextLoader:
    def test_load_single_file(self, tmp_path: Path) -> None:
        f = tmp_path / "note.txt"
        f.write_text("Patient presents with chest pain.")

        loader = PlainTextLoader(note_type="Progress Note")
        records = loader.load_all(str(f))

        assert len(records) == 1
        assert records[0].raw_text == "Patient presents with chest pain."
        assert records[0].note_type == "Progress Note"
        assert records[0].source == "plain_text"

    def test_load_directory(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("Note A content.")
        (tmp_path / "b.txt").write_text("Note B content.")
        (tmp_path / "c.csv").write_text("not loaded")  # should be ignored

        loader = PlainTextLoader()
        records = loader.load_all(str(tmp_path))

        assert len(records) == 2
        texts = {r.raw_text for r in records}
        assert texts == {"Note A content.", "Note B content."}

    def test_load_missing_source_raises(self) -> None:
        loader = PlainTextLoader()
        with pytest.raises(LoaderError, match="not found"):
            loader.load_all("/nonexistent/path/file.txt")

    def test_each_record_has_unique_id(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("A")
        (tmp_path / "b.txt").write_text("B")

        loader = PlainTextLoader()
        records = loader.load_all(str(tmp_path))
        ids = {r.record_id for r in records}
        assert len(ids) == 2


class TestMIMICNotesLoader:
    def _write_mimic_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["note_id", "text", "note_type", "CHARTDATE"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_load_basic_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "notes.csv"
        self._write_mimic_csv(csv_path, [
            {"note_id": "1", "text": "Discharge note text.", "note_type": "Discharge summary", "CHARTDATE": "2023-01-01"},
            {"note_id": "2", "text": "Radiology note text.", "note_type": "Radiology", "CHARTDATE": "2023-01-02"},
        ])

        loader = MIMICNotesLoader()
        records = loader.load_all(str(csv_path))

        assert len(records) == 2
        assert records[0].raw_text == "Discharge note text."
        assert records[0].note_type == "Discharge summary"
        assert records[0].source == "mimic"

    def test_record_ids_are_hashed_not_raw(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "notes.csv"
        self._write_mimic_csv(csv_path, [
            {"note_id": "12345", "text": "Some text.", "note_type": "Note", "CHARTDATE": ""},
        ])

        loader = MIMICNotesLoader(hash_salt="test-salt")
        records = loader.load_all(str(csv_path))

        assert records[0].record_id != "12345"
        assert len(records[0].record_id) == 16  # truncated hex hash

    def test_skips_empty_text_rows(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "notes.csv"
        self._write_mimic_csv(csv_path, [
            {"note_id": "1", "text": "", "note_type": "Note", "CHARTDATE": ""},
            {"note_id": "2", "text": "Real content.", "note_type": "Note", "CHARTDATE": ""},
        ])

        loader = MIMICNotesLoader()
        records = loader.load_all(str(csv_path))
        assert len(records) == 1
        assert records[0].raw_text == "Real content."

    def test_missing_file_raises(self) -> None:
        loader = MIMICNotesLoader()
        with pytest.raises(LoaderError, match="not found"):
            loader.load_all("/nonexistent/notes.csv")


class TestCDAClinicalLoader:
    _SAMPLE_CDA = """<?xml version="1.0"?>
<ClinicalDocument xmlns="urn:hl7-org:v3">
  <component>
    <structuredBody>
      <component>
        <section>
          <title>Chief Complaint</title>
          <text>Shortness of breath.</text>
        </section>
      </component>
      <component>
        <section>
          <title>Assessment</title>
          <text>Likely COPD exacerbation.</text>
        </section>
      </component>
    </structuredBody>
  </component>
</ClinicalDocument>
"""

    def test_load_cda_extracts_sections(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.xml"
        f.write_text(self._SAMPLE_CDA)

        loader = CDAClinicalLoader(extract_sections=True)
        records = loader.load_all(str(f))

        assert len(records) == 1
        assert records[0].sections is not None
        assert "Chief Complaint" in records[0].sections
        assert "Shortness of breath" in records[0].sections["Chief Complaint"]
        assert "Assessment" in records[0].sections

    def test_load_cda_without_section_extraction(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.xml"
        f.write_text(self._SAMPLE_CDA)

        loader = CDAClinicalLoader(extract_sections=False)
        records = loader.load_all(str(f))

        assert records[0].sections is None
        assert "Shortness of breath" in records[0].raw_text

    def test_invalid_xml_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.xml"
        f.write_text("<not><valid<xml")

        loader = CDAClinicalLoader()
        with pytest.raises(LoaderError):
            loader.load_all(str(f))
