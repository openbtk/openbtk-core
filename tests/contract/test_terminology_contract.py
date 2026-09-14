"""Shared contract every registered BaseTerminologyService must satisfy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from openbtk.core.registry import TERMINOLOGY_REGISTRY
from openbtk.core.schemas import CodeSystem, Concept

if TYPE_CHECKING:
    from openbtk.core.base import BaseTerminologyService


def _new_instance(key: str) -> BaseTerminologyService:
    return TERMINOLOGY_REGISTRY.create(key)


# A code guaranteed not to exist in any real or reference vocabulary.
_NONEXISTENT_CODE = "ZZZ-DOES-NOT-EXIST-99999"


@pytest.mark.parametrize("key", TERMINOLOGY_REGISTRY.list_keys())
class TestTerminologyContract:
    def test_resolve_unknown_code_returns_none_not_raise(self, key: str) -> None:
        """An unknown code resolves to None -- it is not an error to ask
        about a code that doesn't exist in this backend."""
        service = _new_instance(key)
        result = service.resolve(_NONEXISTENT_CODE, CodeSystem.SNOMED)
        assert result is None

    def test_validate_unknown_code_is_false(self, key: str) -> None:
        service = _new_instance(key)
        assert service.validate(_NONEXISTENT_CODE, CodeSystem.SNOMED) is False

    def test_resolve_and_validate_agree(self, key: str) -> None:
        """If resolve() finds a concept, validate() for the same code and
        system must agree it's valid -- and vice versa. These must never
        contradict each other."""
        service = _new_instance(key)
        for code, system in [(_NONEXISTENT_CODE, CodeSystem.SNOMED)]:
            resolved = service.resolve(code, system)
            valid = service.validate(code, system)
            assert (resolved is not None) == valid

    def test_map_returns_a_list_of_concepts(self, key: str) -> None:
        service = _new_instance(key)
        result = service.map(_NONEXISTENT_CODE, CodeSystem.SNOMED, CodeSystem.LOINC)
        assert isinstance(result, list)
        assert all(isinstance(c, Concept) for c in result)

    def test_provenance_is_serialisable(self, key: str) -> None:
        service = _new_instance(key)
        dumped = service.provenance().model_dump_json()
        assert isinstance(dumped, str) and len(dumped) > 0
