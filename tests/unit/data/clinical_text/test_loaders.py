"""Unit tests for openbtk.data.clinical_text.loaders.

Covers what tests/contract/test_loader_contract.py deliberately does not:
loader-specific bad-source behaviour, missing-dependency naming, and real
per-loader laziness (see that file's module docstring for why these are
not generically parametrizable).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from openbtk.core.errors import LoaderError, MissingDependencyError
from openbtk.data.clinical_text import loaders
from openbtk.data.clinical_text.loaders import (
    JSONLLoader,
    MIMICNotesLoader,
    PlainTextLoader,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestPlainTextLoader:
    def test_loads_every_txt_file_in_sorted_order(self, tmp_path: Path) -> None:
        (tmp_path / "b.txt").write_text("second")
        (tmp_path / "a.txt").write_text("first")
        (tmp_path / "ignored.md").write_text("not a note")
        records = list(PlainTextLoader().load(str(tmp_path)))
        assert [r.record_id for r in records] == ["a", "b"]
        assert [r.text for r in records] == ["first", "second"]

    def test_note_type_is_applied_to_every_record(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("x")
        records = list(
            PlainTextLoader(note_type="Discharge Summary").load(str(tmp_path))
        )
        assert records[0].note_type == "Discharge Summary"

    def test_source_field_is_plain_text(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("x")
        records = list(PlainTextLoader().load(str(tmp_path)))
        assert records[0].source == "plain_text"

    def test_non_directory_source_raises_loader_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist"
        with pytest.raises(LoaderError, match="Not a directory"):
            list(PlainTextLoader().load(str(missing)))

    def test_undecodable_file_raises_loader_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.txt"
        path.write_bytes(b"\xff\xfe\x00\x81")  # not valid ascii
        with pytest.raises(LoaderError, match="Failed to read"):
            list(PlainTextLoader(encoding="ascii").load(str(tmp_path)))

    def test_empty_directory_yields_nothing(self, tmp_path: Path) -> None:
        assert list(PlainTextLoader().load(str(tmp_path))) == []

    def test_first_record_is_yielded_without_reading_a_later_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Real per-file laziness: file content for "b.txt" must not be
        read before "a.txt"'s record is yielded."""
        (tmp_path / "a.txt").write_text("first")
        (tmp_path / "b.txt").write_text("second")

        from pathlib import Path as RealPath

        original_read_text = RealPath.read_text

        def guarded_read_text(self: RealPath, *args: object, **kwargs: object) -> str:
            if self.name == "b.txt":
                raise AssertionError("b.txt was read before a.txt's record was yielded")
            return original_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(RealPath, "read_text", guarded_read_text)
        first = next(iter(PlainTextLoader().load(str(tmp_path))))
        assert first.text == "first"


class TestJSONLLoader:
    def test_loads_every_line(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.jsonl"
        path.write_text(
            '{"record_id": "n1", "text": "first"}\n'
            '{"record_id": "n2", "text": "second"}\n'
        )
        records = list(JSONLLoader().load(str(path)))
        assert [r.record_id for r in records] == ["n1", "n2"]

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.jsonl"
        path.write_text('{"record_id": "n1", "text": "x"}\n\n   \n')
        records = list(JSONLLoader().load(str(path)))
        assert len(records) == 1

    def test_default_source_is_jsonl_when_absent(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.jsonl"
        path.write_text('{"record_id": "n1", "text": "x"}\n')
        records = list(JSONLLoader().load(str(path)))
        assert records[0].source == "jsonl"

    def test_explicit_source_in_the_line_is_respected(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.jsonl"
        path.write_text('{"record_id": "n1", "text": "x", "source": "synthea"}\n')
        records = list(JSONLLoader().load(str(path)))
        assert records[0].source == "synthea"

    def test_missing_file_raises_loader_error(self, tmp_path: Path) -> None:
        with pytest.raises(LoaderError, match="Could not open"):
            list(JSONLLoader().load(str(tmp_path / "missing.jsonl")))

    def test_invalid_json_names_the_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.jsonl"
        path.write_text('{"record_id": "n1", "text": "ok"}\nnot json\n')
        with pytest.raises(LoaderError) as exc_info:
            list(JSONLLoader().load(str(path)))
        assert exc_info.value.context["line"] == 2

    def test_schema_violation_raises_loader_error(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.jsonl"
        path.write_text('{"text": "missing record_id"}\n')
        with pytest.raises(LoaderError, match="not a valid ClinicalTextRecord"):
            list(JSONLLoader().load(str(path)))

    def test_first_record_is_yielded_without_parsing_a_later_bad_line(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "notes.jsonl"
        path.write_text('{"record_id": "n1", "text": "first"}\nnot json at all\n')
        first = next(iter(JSONLLoader().load(str(path))))
        assert first.text == "first"


class TestMIMICNotesLoader:
    def test_loads_every_row(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.csv"
        path.write_text("ROW_ID,TEXT\n1,first\n2,second\n")
        records = list(MIMICNotesLoader().load(str(path)))
        assert [r.text for r in records] == ["first", "second"]
        assert [r.record_id for r in records] == ["1", "2"]

    def test_source_field_is_mimic_iv_note(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.csv"
        path.write_text("ROW_ID,TEXT\n1,x\n")
        records = list(MIMICNotesLoader().load(str(path)))
        assert records[0].source == "mimic-iv-note"

    def test_optional_columns_map_correctly(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.csv"
        path.write_text(
            "ROW_ID,TEXT,CATEGORY,SUBJECT_ID,HADM_ID\n1,x,Radiology,42,99\n"
        )
        record = next(iter(MIMICNotesLoader().load(str(path))))
        assert record.note_type == "Radiology"
        assert record.patient_ref == "42"
        assert record.encounter_ref == "99"

    def test_missing_optional_columns_become_none(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.csv"
        path.write_text("ROW_ID,TEXT\n1,x\n")
        record = next(iter(MIMICNotesLoader().load(str(path))))
        assert record.note_type is None
        assert record.patient_ref is None
        assert record.encounter_ref is None

    def test_blank_optional_cell_becomes_none_not_nan_string(
        self, tmp_path: Path
    ) -> None:
        """pandas represents a missing CSV cell as float('nan'), not None
        or an empty string -- a real, easy-to-miss gotcha this test guards
        against turning into str(nan) == "nan" leaking into a record."""
        path = tmp_path / "notes.csv"
        path.write_text("ROW_ID,TEXT,CATEGORY\n1,x,\n")
        record = next(iter(MIMICNotesLoader().load(str(path))))
        assert record.note_type is None

    def test_streams_across_multiple_chunks_in_order(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.csv"
        path.write_text("ROW_ID,TEXT\n1,first\n2,second\n3,third\n")
        records = list(MIMICNotesLoader(chunk_size=1).load(str(path)))
        assert [r.text for r in records] == ["first", "second", "third"]

    def test_missing_required_column_raises_loader_error(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.csv"
        path.write_text("ROW_ID,NOT_TEXT\n1,x\n")
        with pytest.raises(LoaderError, match="missing required column"):
            list(MIMICNotesLoader().load(str(path)))

    def test_missing_file_raises_loader_error(self, tmp_path: Path) -> None:
        with pytest.raises(LoaderError, match="Could not read"):
            list(MIMICNotesLoader().load(str(tmp_path / "missing.csv")))

    def test_missing_pandas_raises_missing_dependency_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Doesn't need pandas to genuinely be uninstalled -- monkeypatches
        require() to fail exactly the way it does when the "text" extra
        isn't installed."""

        def fake_require(module: str, *, extra: str) -> None:
            raise MissingDependencyError(
                f"'{module}' is required but not installed.",
                context={"module": module, "extra": extra},
            )

        monkeypatch.setattr(loaders, "require", fake_require)
        path = tmp_path / "notes.csv"
        path.write_text("ROW_ID,TEXT\n1,x\n")
        with pytest.raises(MissingDependencyError) as exc_info:
            list(MIMICNotesLoader().load(str(path)))
        assert exc_info.value.context["extra"] == "text"
