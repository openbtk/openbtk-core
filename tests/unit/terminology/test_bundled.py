"""Unit tests for openbtk.terminology.bundled.BundledMinimalBackend."""

from __future__ import annotations

from openbtk.core.schemas import CodeSystem, Concept
from openbtk.terminology.bundled import BundledMinimalBackend


class TestResolve:
    def test_resolves_a_known_code(self) -> None:
        backend = BundledMinimalBackend()
        concept = backend.resolve("E11.9", CodeSystem.ICD10CM)
        assert concept == Concept(
            code="E11.9",
            system=CodeSystem.ICD10CM,
            display="Type 2 diabetes mellitus without complications",
        )

    def test_unknown_code_returns_none(self) -> None:
        backend = BundledMinimalBackend()
        assert backend.resolve("Z99.999", CodeSystem.ICD10CM) is None

    def test_only_icd10cm_is_supported(self) -> None:
        backend = BundledMinimalBackend()
        assert backend.resolve("E11.9", CodeSystem.SNOMED) is None


class TestValidate:
    def test_true_for_a_known_code(self) -> None:
        backend = BundledMinimalBackend()
        assert backend.validate("I10", CodeSystem.ICD10CM) is True

    def test_false_for_an_unknown_code(self) -> None:
        backend = BundledMinimalBackend()
        assert backend.validate("bogus", CodeSystem.ICD10CM) is False


class TestAuthority:
    def test_the_subset_is_never_authoritative_for_any_system(self) -> None:
        backend = BundledMinimalBackend()
        assert all(not backend.is_authoritative(s) for s in CodeSystem)


class TestMap:
    def test_always_returns_empty_list(self) -> None:
        backend = BundledMinimalBackend()
        result = backend.map("E11.9", CodeSystem.ICD10CM, CodeSystem.SNOMED)
        assert result == []


class TestRegistration:
    def test_registered_under_the_expected_key(self) -> None:
        assert (
            BundledMinimalBackend.registry_key == "terminology.general.bundled_minimal"
        )

    def test_provenance_is_serialisable(self) -> None:
        backend = BundledMinimalBackend()
        assert isinstance(backend.provenance().model_dump_json(), str)
