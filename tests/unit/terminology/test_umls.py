"""Unit tests for openbtk.terminology.umls.UMLSRestBackend.

Built directly on httpx (a core dependency), so these tests use a real
httpx.MockTransport rather than mocking an SDK -- the same convention
tests/unit/llms/test_openai_compatible.py already established for the
project's other httpx-native provider.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from openbtk.core.errors import TerminologyError
from openbtk.core.schemas import CodeSystem, Concept
from openbtk.terminology.umls import UMLSRestBackend


def _backend_with_transport(handler: Any) -> UMLSRestBackend:
    backend = UMLSRestBackend(api_key="secret-key")  # pragma: allowlist secret
    client = backend._get_client()
    client._transport = httpx.MockTransport(handler)
    return backend


def _search_response(results: list[dict[str, Any]]) -> httpx.Response:
    return httpx.Response(200, json={"result": {"results": results}})


class TestResolve:
    def test_returns_a_concept_when_the_code_matches(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "apiKey=secret-key" in str(request.url)
            assert "sabs=SNOMEDCT_US" in str(request.url)
            return _search_response(
                [{"ui": "385093006", "name": "Community-acquired pneumonia"}]
            )

        backend = _backend_with_transport(handler)
        concept = backend.resolve("385093006", CodeSystem.SNOMED)
        assert concept == Concept(
            code="385093006",
            system=CodeSystem.SNOMED,
            display="Community-acquired pneumonia",
        )

    def test_returns_none_when_no_result_matches(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _search_response([])

        backend = _backend_with_transport(handler)
        assert backend.resolve("nope", CodeSystem.SNOMED) is None

    def test_returns_none_when_ui_does_not_match_the_queried_code(self) -> None:
        """A defensive check: only an exact ui match counts, even if the
        API returned other, unrelated results."""

        def handler(request: httpx.Request) -> httpx.Response:
            return _search_response(
                [{"ui": "different-code", "name": "Something else"}]
            )

        backend = _backend_with_transport(handler)
        assert backend.resolve("385093006", CodeSystem.SNOMED) is None

    def test_uses_the_correct_sab_per_system(self) -> None:
        seen_sabs = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_sabs.append(request.url.params["sabs"])
            return _search_response([])

        backend = _backend_with_transport(handler)
        backend.resolve("x", CodeSystem.LOINC)
        backend.resolve("x", CodeSystem.ICD10CM)
        backend.resolve("x", CodeSystem.RXNORM)
        backend.resolve("x", CodeSystem.CPT)
        assert seen_sabs == ["LNC", "ICD10CM", "RXNORM", "CPT"]

    def test_ucum_is_not_supported(self) -> None:
        backend = UMLSRestBackend(api_key="x")  # pragma: allowlist secret
        with pytest.raises(TerminologyError, match="does not support UCUM"):
            backend.resolve("x", CodeSystem.UCUM)

    def test_http_error_raises_terminology_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"message": "Invalid API Key"})

        backend = _backend_with_transport(handler)
        with pytest.raises(TerminologyError):
            backend.resolve("x", CodeSystem.SNOMED)

    def test_api_key_never_appears_in_error_context(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"message": "Invalid API Key"})

        backend = _backend_with_transport(handler)
        with pytest.raises(TerminologyError) as exc_info:
            backend.resolve("x", CodeSystem.SNOMED)
        assert "secret-key" not in str(exc_info.value)
        assert "secret-key" not in str(exc_info.value.context)


class TestValidate:
    def test_mirrors_resolve(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _search_response([{"ui": "E11.9", "name": "Diabetes"}])

        backend = _backend_with_transport(handler)
        assert backend.validate("E11.9", CodeSystem.ICD10CM) is True

    def test_false_for_unknown_code(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _search_response([])

        backend = _backend_with_transport(handler)
        assert backend.validate("nope", CodeSystem.ICD10CM) is False


class TestMap:
    def test_maps_through_cui_to_target_system_atoms(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/rest/search/current":
                return _search_response([{"ui": "C0011849", "name": "Diabetes"}])
            assert request.url.path == "/rest/content/current/CUI/C0011849/atoms"
            return httpx.Response(
                200,
                json={
                    "result": [
                        {
                            "code": (
                                "https://uts-ws.nlm.nih.gov/rest/content/current/"
                                "source/RXNORM/860975"
                            ),
                            "name": "metformin 500 MG",
                        }
                    ]
                },
            )

        backend = _backend_with_transport(handler)
        result = backend.map("E11.9", CodeSystem.ICD10CM, CodeSystem.RXNORM)
        assert result == [
            Concept(code="860975", system=CodeSystem.RXNORM, display="metformin 500 MG")
        ]

    def test_no_matching_cui_returns_empty_list(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _search_response([])

        backend = _backend_with_transport(handler)
        assert backend.map("nope", CodeSystem.ICD10CM, CodeSystem.RXNORM) == []

    def test_atom_missing_code_or_name_is_skipped(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/rest/search/current":
                return _search_response([{"ui": "C0011849", "name": "Diabetes"}])
            return httpx.Response(200, json={"result": [{"name": "no code here"}]})

        backend = _backend_with_transport(handler)
        assert backend.map("E11.9", CodeSystem.ICD10CM, CodeSystem.RXNORM) == []

    def test_atoms_http_error_raises_terminology_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/rest/search/current":
                return _search_response([{"ui": "C0011849", "name": "Diabetes"}])
            return httpx.Response(500, json={"message": "server error"})

        backend = _backend_with_transport(handler)
        with pytest.raises(TerminologyError):
            backend.map("E11.9", CodeSystem.ICD10CM, CodeSystem.RXNORM)


class TestDeclaredAttributes:
    def test_sends_data_offsite_is_true(self) -> None:
        assert UMLSRestBackend.sends_data_offsite is True

    def test_registered_under_the_expected_key(self) -> None:
        assert UMLSRestBackend.registry_key == "terminology.general.umls"

    def test_provenance_is_serialisable(self) -> None:
        backend = UMLSRestBackend(api_key="x")  # pragma: allowlist secret
        assert isinstance(backend.provenance().model_dump_json(), str)
