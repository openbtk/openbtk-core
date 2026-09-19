"""Unit tests for openbtk.terminology.local.LocalVocabBackend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from openbtk.core.errors import TerminologyError
from openbtk.core.schemas import CodeSystem, Concept
from openbtk.terminology.local import LocalVocabBackend

if TYPE_CHECKING:
    from pathlib import Path


def _write_vocab(path: Path, rows: list[str]) -> str:
    content = "code,system,display\n" + "\n".join(rows) + "\n"
    path.write_text(content, encoding="utf-8")
    return str(path)


class TestLoading:
    def test_constructing_does_no_io(self, tmp_path: Path) -> None:
        """rule 11: no file read until a real method call."""
        LocalVocabBackend(path=str(tmp_path / "does-not-exist.csv"))

    def test_resolves_a_row(self, tmp_path: Path) -> None:
        path = _write_vocab(
            tmp_path / "vocab.csv", ["73211009,SNOMED,Diabetes mellitus"]
        )
        backend = LocalVocabBackend(path=path)
        assert backend.resolve("73211009", CodeSystem.SNOMED) == Concept(
            code="73211009", system=CodeSystem.SNOMED, display="Diabetes mellitus"
        )

    def test_loads_only_once_across_calls(self, tmp_path: Path) -> None:
        path = _write_vocab(tmp_path / "vocab.csv", ["X,SNOMED,Foo"])
        backend = LocalVocabBackend(path=path)
        backend.resolve("X", CodeSystem.SNOMED)
        (tmp_path / "vocab.csv").write_text("code,system,display\n", encoding="utf-8")
        # Still resolves from the already-loaded, in-memory table.
        assert backend.resolve("X", CodeSystem.SNOMED) is not None

    def test_missing_file_raises_terminology_error(self, tmp_path: Path) -> None:
        backend = LocalVocabBackend(path=str(tmp_path / "nope.csv"))
        with pytest.raises(TerminologyError):
            backend.resolve("X", CodeSystem.SNOMED)

    def test_missing_column_raises_terminology_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        path.write_text("code,display\nX,Foo\n", encoding="utf-8")
        backend = LocalVocabBackend(path=str(path))
        with pytest.raises(TerminologyError, match="missing required column"):
            backend.resolve("X", CodeSystem.SNOMED)

    def test_unknown_system_raises_terminology_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        path.write_text("code,system,display\nX,NOT_A_SYSTEM,Foo\n", encoding="utf-8")
        backend = LocalVocabBackend(path=str(path))
        with pytest.raises(TerminologyError, match="unknown system"):
            backend.resolve("X", CodeSystem.SNOMED)


class TestResolveValidate:
    def test_unknown_code_resolves_to_none(self, tmp_path: Path) -> None:
        path = _write_vocab(tmp_path / "vocab.csv", ["X,SNOMED,Foo"])
        backend = LocalVocabBackend(path=path)
        assert backend.resolve("Y", CodeSystem.SNOMED) is None

    def test_validate_mirrors_resolve(self, tmp_path: Path) -> None:
        path = _write_vocab(tmp_path / "vocab.csv", ["X,SNOMED,Foo"])
        backend = LocalVocabBackend(path=path)
        assert backend.validate("X", CodeSystem.SNOMED) is True
        assert backend.validate("Y", CodeSystem.SNOMED) is False

    def test_multiple_systems_in_one_file(self, tmp_path: Path) -> None:
        path = _write_vocab(
            tmp_path / "vocab.csv",
            ["X,SNOMED,Foo", "X,ICD10CM,Bar"],
        )
        backend = LocalVocabBackend(path=path)
        assert backend.resolve("X", CodeSystem.SNOMED).display == "Foo"  # type: ignore[union-attr]
        assert backend.resolve("X", CodeSystem.ICD10CM).display == "Bar"  # type: ignore[union-attr]


class TestAuthority:
    def test_authoritative_only_for_systems_the_file_has_rows_for(
        self, tmp_path: Path
    ) -> None:
        backend = LocalVocabBackend(
            path=_write_vocab(tmp_path / "v.csv", ["385093006,SNOMED,Pneumonia"])
        )
        assert backend.is_authoritative(CodeSystem.SNOMED) is True
        assert backend.is_authoritative(CodeSystem.LOINC) is False


class TestMap:
    def test_always_returns_empty_list(self, tmp_path: Path) -> None:
        path = _write_vocab(tmp_path / "vocab.csv", ["X,SNOMED,Foo"])
        backend = LocalVocabBackend(path=path)
        assert backend.map("X", CodeSystem.SNOMED, CodeSystem.ICD10CM) == []


class TestRegistration:
    def test_registered_under_the_expected_key(self) -> None:
        assert LocalVocabBackend.registry_key == "terminology.general.local"

    def test_provenance_is_serialisable(self, tmp_path: Path) -> None:
        backend = LocalVocabBackend(path=str(tmp_path / "x.csv"))
        assert isinstance(backend.provenance().model_dump_json(), str)
