"""Shared contract every registered BaseTerminologyService must satisfy.

``terminology.general.umls`` is the first offsite-sending, real-network
terminology backend (task 7.1) -- it needs the same two things the LLM/
embedding contract suites already gate real calls on: an explicit
``OPENBTK_SLOW_TESTS=1`` opt-in, and (since obtaining a real UMLS API key
requires an approved individual licence this environment does not have)
a real credential named by an env var, without which the call would just
fail with a predictable auth error rather than exercising anything this
suite exists to verify. It also needs the policy that permits
constructing an offsite provider at all, the same as every offsite LLM/
embedding provider's own `_new_instance` in their contract suites.
``terminology.general.local`` needs a real, per-key CSV fixture, the same
per-key source-fixture pattern already established for loaders.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest

from openbtk.core.config import PolicyConfig
from openbtk.core.registry import TERMINOLOGY_REGISTRY
from openbtk.core.schemas import CodeSystem, Concept

from .conftest import enrolled

if TYPE_CHECKING:
    from pathlib import Path

    from openbtk.core.base import BaseTerminologyService

_ALLOW_OFFSITE = PolicyConfig(allow_offsite_providers=True)

_UMLS_KEY = "terminology.general.umls"
_UMLS_API_KEY_ENV_VAR = "OPENBTK_TEST_UMLS_API_KEY"  # pragma: allowlist secret


def _constructor_kwargs(key: str, tmp_path: Path) -> dict[str, Any]:
    if key == _UMLS_KEY:
        return {"api_key": os.environ.get(_UMLS_API_KEY_ENV_VAR, "unused-in-ci")}
    if key == "terminology.general.local":
        path = tmp_path / "vocab.csv"
        path.write_text(
            "code,system,display\nplaceholder,SNOMED,Placeholder\n", encoding="utf-8"
        )
        return {"path": str(path)}
    return {}


def _new_instance(key: str, tmp_path: Path) -> BaseTerminologyService:
    return TERMINOLOGY_REGISTRY.create(
        key, policy=_ALLOW_OFFSITE, **_constructor_kwargs(key, tmp_path)
    )


def _skip_if_real_call_unavailable(key: str) -> None:
    if key != _UMLS_KEY:
        return
    if os.environ.get("OPENBTK_SLOW_TESTS") != "1":
        pytest.skip(f"{key}: makes a real network call; set OPENBTK_SLOW_TESTS=1")
    if not os.environ.get(_UMLS_API_KEY_ENV_VAR):
        pytest.skip(f"{key}: requires {_UMLS_API_KEY_ENV_VAR} for a real call")


# A code guaranteed not to exist in any real or reference vocabulary.
_NONEXISTENT_CODE = "ZZZ-DOES-NOT-EXIST-99999"


@pytest.mark.parametrize("key", enrolled(TERMINOLOGY_REGISTRY))
class TestTerminologyContract:
    def test_resolve_unknown_code_returns_none_not_raise(
        self, key: str, tmp_path: Path
    ) -> None:
        """An unknown code resolves to None -- it is not an error to ask
        about a code that doesn't exist in this backend."""
        _skip_if_real_call_unavailable(key)
        service = _new_instance(key, tmp_path)
        result = service.resolve(_NONEXISTENT_CODE, CodeSystem.SNOMED)
        assert result is None

    def test_validate_unknown_code_is_false(self, key: str, tmp_path: Path) -> None:
        _skip_if_real_call_unavailable(key)
        service = _new_instance(key, tmp_path)
        assert service.validate(_NONEXISTENT_CODE, CodeSystem.SNOMED) is False

    def test_resolve_and_validate_agree(self, key: str, tmp_path: Path) -> None:
        """If resolve() finds a concept, validate() for the same code and
        system must agree it's valid -- and vice versa. These must never
        contradict each other."""
        _skip_if_real_call_unavailable(key)
        service = _new_instance(key, tmp_path)
        for code, system in [(_NONEXISTENT_CODE, CodeSystem.SNOMED)]:
            resolved = service.resolve(code, system)
            valid = service.validate(code, system)
            assert (resolved is not None) == valid

    def test_is_authoritative_is_a_bool_for_every_system(
        self, key: str, tmp_path: Path
    ) -> None:
        """Says whether validate()'s False means "does not exist" or "cannot
        confirm", so a guardrail never calls a valid code invalid."""
        service = _new_instance(key, tmp_path)
        assert all(isinstance(service.is_authoritative(s), bool) for s in CodeSystem)

    def test_map_returns_a_list_of_concepts(self, key: str, tmp_path: Path) -> None:
        _skip_if_real_call_unavailable(key)
        service = _new_instance(key, tmp_path)
        result = service.map(_NONEXISTENT_CODE, CodeSystem.SNOMED, CodeSystem.LOINC)
        assert isinstance(result, list)
        assert all(isinstance(c, Concept) for c in result)

    def test_provenance_is_serialisable(self, key: str, tmp_path: Path) -> None:
        service = _new_instance(key, tmp_path)
        dumped = service.provenance().model_dump_json()
        assert isinstance(dumped, str) and len(dumped) > 0
