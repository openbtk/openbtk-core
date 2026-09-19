"""Unit tests for openbtk.data.clinical_text.datasets.N2C2DeidDataset.

Every input is a hand-built file in the i2b2 2014 ``<deIdi2b2>`` shape --
never real data (the real corpus is DUA-restricted and this environment does
not have it; the adapter's own docstring says so).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from openbtk.core.errors import DatasetError
from openbtk.data.clinical_text.datasets import _TAG_TO_CATEGORY, N2C2DeidDataset
from openbtk.deid.schemas import PHICategory

if TYPE_CHECKING:
    from pathlib import Path

_TEXT = "Patient Jane Roe seen 2090-01-05 at Mercy Hospital, age 45."


def _tag(
    element: str, typ: str, start: int, end: int, *, text: str | None = None
) -> str:
    claimed = _TEXT[start:end] if text is None else text
    return (
        f'<{element} id="P0" start="{start}" end="{end}" text="{claimed}" '
        f'TYPE="{typ}" comment=""/>'
    )


def _doc(tags: str, text: str = _TEXT) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?><deIdi2b2>'
        f"<TEXT><![CDATA[{text}]]></TEXT><TAGS>{tags}</TAGS></deIdi2b2>"
    )


def _write(directory: Path, name: str, content: str) -> None:
    (directory / name).write_text(content, encoding="utf-8")


class TestRefusesWithoutData:
    def test_no_path_raises_and_names_the_registration_page(self) -> None:
        with pytest.raises(DatasetError, match="Register at https://") as exc:
            N2C2DeidDataset().load()
        assert "never downloaded" in str(exc.value)
        assert "registration" in exc.value.context

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="Register at"):
            N2C2DeidDataset(path=str(tmp_path / "nope")).load()

    def test_a_file_instead_of_a_directory_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "x.xml"
        f.write_text("<a/>", encoding="utf-8")
        with pytest.raises(DatasetError):
            N2C2DeidDataset(path=str(f)).load()

    def test_refuses_at_load_call_not_at_first_next(self) -> None:
        """load() validates eagerly: the failure must not hide until a
        caller happens to iterate."""
        with pytest.raises(DatasetError):
            N2C2DeidDataset().load()


class TestDeclaredAttributes:
    def test_requires_credentials(self) -> None:
        assert N2C2DeidDataset().requires_credentials is True

    def test_license_is_the_registration_url(self) -> None:
        assert N2C2DeidDataset().license.startswith("https://")

    def test_name_is_nonempty(self) -> None:
        assert N2C2DeidDataset().name

    def test_registered_under_the_expected_key(self) -> None:
        assert N2C2DeidDataset.registry_key == "dataset.clinical_text.n2c2_deid"


class TestParsing:
    def test_maps_tags_to_safe_harbor_categories_with_offsets(
        self, tmp_path: Path
    ) -> None:
        tags = _tag("NAME", "PATIENT", 8, 16) + _tag("DATE", "DATE", 22, 32)
        _write(tmp_path, "doc1.xml", _doc(tags))
        (doc,) = list(N2C2DeidDataset(path=str(tmp_path)).load())
        assert doc.document_id == "doc1"
        assert doc.text == _TEXT
        assert [(s.category, s.start, s.end) for s in doc.spans] == [
            (PHICategory.NAME, 8, 16),
            (PHICategory.DATE, 22, 32),
        ]
        assert doc.unscored_spans == 0

    def test_annotations_without_a_safe_harbor_category_are_counted_not_dropped(
        self, tmp_path: Path
    ) -> None:
        tags = (
            _tag("NAME", "PATIENT", 8, 16)
            + _tag("LOCATION", "HOSPITAL", 36, 50)
            + _tag("AGE", "AGE", 56, 58)
            + _tag("PROFESSION", "PROFESSION", 0, 7)
            + _tag("ID", "SOMETHING_NEW", 0, 3)
        )
        _write(tmp_path, "d.xml", _doc(tags))
        (doc,) = list(N2C2DeidDataset(path=str(tmp_path)).load())
        assert len(doc.spans) == 1
        assert doc.unscored_spans == 4

    def test_path_may_be_given_to_load_instead_of_the_constructor(
        self, tmp_path: Path
    ) -> None:
        _write(tmp_path, "d.xml", _doc(""))
        docs = list(N2C2DeidDataset().load(path=str(tmp_path)))
        assert len(docs) == 1

    def test_documents_stream_in_sorted_filename_order(self, tmp_path: Path) -> None:
        for name in ("b.xml", "a.xml", "c.xml"):
            _write(tmp_path, name, _doc(""))
        ids = [d.document_id for d in N2C2DeidDataset(path=str(tmp_path)).load()]
        assert ids == ["a", "b", "c"]

    def test_only_xml_files_are_read(self, tmp_path: Path) -> None:
        _write(tmp_path, "a.xml", _doc(""))
        (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
        assert len(list(N2C2DeidDataset(path=str(tmp_path)).load())) == 1

    def test_a_document_with_no_tags_element_has_no_spans(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            "d.xml",
            f"<deIdi2b2><TEXT><![CDATA[{_TEXT}]]></TEXT></deIdi2b2>",
        )
        (doc,) = list(N2C2DeidDataset(path=str(tmp_path)).load())
        assert doc.spans == []

    def test_a_tag_without_a_text_attribute_is_accepted_unverified(
        self, tmp_path: Path
    ) -> None:
        tag = '<NAME id="P0" start="8" end="16" TYPE="PATIENT"/>'
        _write(tmp_path, "d.xml", _doc(tag))
        (doc,) = list(N2C2DeidDataset(path=str(tmp_path)).load())
        assert len(doc.spans) == 1

    def test_streaming_defers_a_later_files_failure(self, tmp_path: Path) -> None:
        _write(tmp_path, "a.xml", _doc(""))
        _write(tmp_path, "b.xml", "<not-xml")
        it = iter(N2C2DeidDataset(path=str(tmp_path)).load())
        assert next(it).document_id == "a"
        with pytest.raises(DatasetError):
            next(it)


class TestMalformedInput:
    def _load_one(self, tmp_path: Path, content: str) -> None:
        _write(tmp_path, "bad.xml", content)
        list(N2C2DeidDataset(path=str(tmp_path)).load())

    def test_misaligned_offsets_are_refused_not_scored(self, tmp_path: Path) -> None:
        tags = _tag("NAME", "PATIENT", 8, 16, text="Someone Else")
        with pytest.raises(DatasetError, match="misaligned") as exc:
            self._load_one(tmp_path, _doc(tags))
        assert exc.value.context["filename"] == "bad.xml"
        # The error names the tag and offsets, never the document's text.
        assert "Jane Roe" not in str(exc.value)
        assert "Jane Roe" not in str(exc.value.context)

    def test_missing_start_or_end_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="integer start/end"):
            self._load_one(tmp_path, _doc('<NAME TYPE="PATIENT" end="3"/>'))

    def test_non_integer_offsets_are_refused(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="integer start/end"):
            self._load_one(tmp_path, _doc('<NAME TYPE="PATIENT" start="a" end="3"/>'))

    def test_malformed_xml_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="well-formed"):
            self._load_one(tmp_path, "<deIdi2b2><TEXT>")

    def test_missing_text_element_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="no <TEXT>"):
            self._load_one(tmp_path, "<deIdi2b2><TAGS/></deIdi2b2>")

    def test_doctype_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="DOCTYPE/ENTITY"):
            self._load_one(tmp_path, '<!DOCTYPE x [<!ENTITY a "b">]><deIdi2b2/>')

    def test_undecodable_bytes_are_refused(self, tmp_path: Path) -> None:
        (tmp_path / "bad.xml").write_bytes(b"\xff\xfe\x00bad")
        with pytest.raises(DatasetError, match="Could not read"):
            list(N2C2DeidDataset(path=str(tmp_path)).load())


class TestMappingTable:
    def test_every_mapped_category_is_a_real_phi_category(self) -> None:
        assert all(isinstance(c, PHICategory) for c in _TAG_TO_CATEGORY.values())

    def test_person_names_all_map_to_name(self) -> None:
        for typ in ("PATIENT", "DOCTOR", "USERNAME"):
            assert _TAG_TO_CATEGORY[("NAME", typ)] is PHICategory.NAME


class TestProvenance:
    def test_provenance_is_serialisable(self) -> None:
        assert isinstance(N2C2DeidDataset().provenance().model_dump_json(), str)
