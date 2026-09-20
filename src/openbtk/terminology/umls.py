"""``UMLSRestBackend``: the real UMLS Terminology Services (UTS) REST API
(docs/03_ARCHITECTURE.md section 8.2), requiring a user's own UMLS
licence and API key -- never bundled, never assumed.

**Real, verified endpoint behaviour, not assumed:**

* Authentication is a single ``?apiKey=<key>`` query parameter appended to
  every request -- confirmed directly against the live service
  (``GET https://uts-ws.nlm.nih.gov/rest/content/current/CUI/C0011849
  ?apiKey=test`` returns a real, well-formed
  ``{"name":"UnauthorizedError","status":401,...}`` naming exactly this
  parameter) and against NLM's own current authentication documentation
  (``https://documentation.uts.nlm.nih.gov/rest/authentication.html``),
  which explicitly deprecates the older ticket-granting-ticket/service-
  ticket flow in favour of this simpler one.
* ``resolve``/``validate`` use the real ``/search/current`` endpoint with
  ``inputType=sourceUi&searchType=exact&returnIdType=code`` -- confirmed
  against NLM's own documented sample URIs
  (``https://documentation.uts.nlm.nih.gov/rest/search/index.html``),
  which return source-vocabulary-native codes and their display names for
  an exact source-code lookup, in one call.
* ``map`` first resolves ``code`` to its UMLS CUI(s) (the same
  ``/search/current`` endpoint, default ``returnIdType=concept``), then
  retrieves that CUI's atoms restricted to the target vocabulary via
  ``/content/current/CUI/{cui}/atoms?sabs=<TO_SAB>`` (NLM's documented
  "Retrieving UMLS Atoms" endpoint). This one is real and documented, but
  -- disclosed honestly -- has not been exercised against a live, licensed
  account in this repository: obtaining a real UMLS API key requires an
  approved individual licence this environment does not have. Every
  method here is instead verified against realistic, hand-built mock
  responses (tests/unit/terminology/test_umls.py), matching this
  project's existing precedent for cloud providers it cannot call live in
  CI either (OpenAI, Anthropic).

**Source-vocabulary abbreviations** (the ``sabs`` parameter) are UMLS's
own fixed, standard abbreviations, not invented: ``SNOMEDCT_US``, ``LNC``
(LOINC), ``ICD10CM``, ``RXNORM``, ``CPT``. ``CodeSystem.UCUM`` is
deliberately unsupported here -- UCUM is a units-of-measure standard, and
this project has not confirmed it is loaded as a standard UMLS source
vocabulary; guessing an abbreviation would risk silently validating
against the wrong (or no) data, which docs/03_ARCHITECTURE.md section 8.2
explicitly warns against ("does not silently return valid").
"""

from __future__ import annotations

from typing import Any, ClassVar

import httpx

from openbtk.core.base import BaseTerminologyService
from openbtk.core.errors import TerminologyError
from openbtk.core.registry import TERMINOLOGY_REGISTRY
from openbtk.core.schemas import CodeSystem, Concept

_BASE_URL = "https://uts-ws.nlm.nih.gov/rest"

# Real, standard UMLS source-vocabulary abbreviations -- not fabricated.
_SAB_BY_CODE_SYSTEM: dict[CodeSystem, str] = {
    CodeSystem.SNOMED: "SNOMEDCT_US",
    CodeSystem.LOINC: "LNC",
    CodeSystem.ICD10CM: "ICD10CM",
    CodeSystem.RXNORM: "RXNORM",
    CodeSystem.CPT: "CPT",
}


@TERMINOLOGY_REGISTRY.register("terminology.general.umls")
class UMLSRestBackend(BaseTerminologyService):
    """Resolve/validate/map codes via the real UMLS REST API.

    Args:
        api_key: A UMLS Terminology Services API key (requires a UMLS
            account and licence -- https://uts.nlm.nih.gov). Never
            defaulted or bundled.
        timeout: Per-request timeout, seconds.

    No I/O happens in ``__init__`` -- the ``httpx.Client`` is built lazily
    on first real call (docs/09_CODING_STANDARDS.md rule 11).
    """

    sends_data_offsite: ClassVar[bool] = True

    def __init__(self, *, api_key: str, timeout: float = 10.0) -> None:
        self._api_key = api_key
        self._timeout = timeout
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=_BASE_URL, timeout=self._timeout)
        return self._client

    def _sab(self, system: CodeSystem) -> str:
        sab = _SAB_BY_CODE_SYSTEM.get(system)
        if sab is None:
            raise TerminologyError(
                f"UMLSRestBackend does not support {system.value}: no "
                "confirmed UMLS source-vocabulary abbreviation for it.",
                context={"system": system.value},
            )
        return sab

    def _search(self, **params: Any) -> list[dict[str, Any]]:
        try:
            response = self._get_client().get(
                "/search/current", params={**params, "apiKey": self._api_key}
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise TerminologyError(
                "UMLS search request failed.", context={"params": _redact(params)}
            ) from _sanitised(e)
        result: list[dict[str, Any]] = (
            response.json().get("result", {}).get("results", [])
        )
        return result

    def resolve(self, code: str, system: CodeSystem) -> Concept | None:
        sab = self._sab(system)
        results = self._search(
            string=code,
            inputType="sourceUi",
            searchType="exact",
            sabs=sab,
            returnIdType="code",
        )
        for item in results:
            if item.get("ui") == code:
                return Concept(code=code, system=system, display=item.get("name", code))
        return None

    def validate(self, code: str, system: CodeSystem) -> bool:
        return self.resolve(code, system) is not None

    def map(
        self, code: str, from_system: CodeSystem, to_system: CodeSystem
    ) -> list[Concept]:
        from_sab = self._sab(from_system)
        to_sab = self._sab(to_system)
        cuis = [
            item["ui"]
            for item in self._search(
                string=code, inputType="sourceUi", searchType="exact", sabs=from_sab
            )
            if "ui" in item
        ]
        concepts: list[Concept] = []
        for cui in cuis:
            concepts.extend(self._atoms_in_system(cui, to_sab, to_system))
        return concepts

    def _atoms_in_system(self, cui: str, sab: str, system: CodeSystem) -> list[Concept]:
        try:
            response = self._get_client().get(
                f"/content/current/CUI/{cui}/atoms",
                params={"sabs": sab, "apiKey": self._api_key},
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise TerminologyError(
                "UMLS atoms request failed.", context={"cui": cui, "sab": sab}
            ) from _sanitised(e)
        atoms = response.json().get("result", [])
        concepts: list[Concept] = []
        for atom in atoms:
            code = atom.get("code", "")
            # UMLS atom "code" fields are full resource URIs
            # (".../source/<SAB>/<CODE>"); the bare code is the final
            # path segment.
            bare_code = code.rsplit("/", 1)[-1] if code else None
            name = atom.get("name")
            if bare_code and name:
                concepts.append(Concept(code=bare_code, system=system, display=name))
        return concepts


def _sanitised(error: httpx.HTTPError) -> Exception:
    """A replacement for ``error`` that cannot carry the API key.

    UMLS takes the key as an ``apiKey`` query parameter, and httpx puts the full
    request URL -- key included -- in its error text ("Client error '401' for url
    '...?apiKey=SECRET'"). Chaining the original exception (``raise ... from e``)
    would print that URL in every traceback, and so in CI logs and crash reports.
    Only the exception type and HTTP status are kept.
    """
    response = getattr(error, "response", None)
    status = f" (HTTP {response.status_code})" if response is not None else ""
    return RuntimeError(f"{type(error).__name__}{status}")


def _redact(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if k != "apiKey"}
