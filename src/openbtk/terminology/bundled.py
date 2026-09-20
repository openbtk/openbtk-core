"""``BundledMinimalBackend``: a small, permissively-licensed terminology
subset shipped with the package, working fully offline with zero network
calls and zero API key (docs/03_ARCHITECTURE.md section 8.2).

**ICD-10-CM only, deliberately.** ICD-10-CM is produced by the U.S. CDC/
NCHS and, as a U.S. government work, is public domain domestically (17
U.S.C. section 105) -- safe to bundle. SNOMED CT is NOT bundled: its
content (concept ids and descriptions) is restricted, licensed vocabulary
(SNOMED International/IHTSDO), and docs/03_ARCHITECTURE.md section 8.2 is
explicit that this project "never bundles restricted vocabularies." LOINC
is also NOT bundled: Regenstrief's LOINC is free to use but redistribution
requires accepting LOINC's own license terms, which this project cannot
attest to on a user's behalf. This is the same "wrap, don't reinvent, and
never fake a license you don't have" discipline already applied to
``openbtk.deid`` (docs/09_CODING_STANDARDS.md section 7).

**Every code below was verified directly** against the real NLM Clinical
Table Search Service (``https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search``,
a free, public, no-API-key NIH/NLM service) at the time this module was
written -- not recalled from training data and not invented (CLAUDE.md rule
14, extended to "no code without a verified source" the same way M5
extended it to model identifiers). This is a small, illustrative,
representative subset spanning common conditions across many ICD-10-CM
chapters -- not remotely exhaustive, and not a substitute for a real
ICD-10-CM release or a licensed backend for anything beyond quick,
offline, common-case validation.
"""

from __future__ import annotations

from openbtk.core.base import BaseTerminologyService
from openbtk.core.registry import TERMINOLOGY_REGISTRY
from openbtk.core.schemas import CodeSystem, Concept

# code -> official short description, each verified against the real NLM
# Clinical Table Search Service (see module docstring).
_ICD10CM_SUBSET: dict[str, str] = {
    "A09": "Infectious gastroenteritis and colitis, unspecified",
    "B34.9": "Viral infection, unspecified",
    "C50.911": "Malignant neoplasm of unspecified site of right female breast",
    "D64.9": "Anemia, unspecified",
    "E11.9": "Type 2 diabetes mellitus without complications",
    "E03.9": "Hypothyroidism, unspecified",
    "F41.1": "Generalized anxiety disorder",
    "G43.909": "Migraine, unspecified, not intractable, without status migrainosus",
    "H52.4": "Presbyopia",
    "H61.23": "Impacted cerumen, bilateral",
    "I10": "Essential (primary) hypertension",
    "I25.10": (
        "Atherosclerotic heart disease of native coronary artery "
        "without angina pectoris"
    ),
    "J18.9": "Pneumonia, unspecified organism",
    "J45.909": "Unspecified asthma, uncomplicated",
    "K21.9": "Gastro-esophageal reflux disease without esophagitis",
    "L20.9": "Atopic dermatitis, unspecified",
    "M54.50": "Low back pain, unspecified",
    "N39.0": "Urinary tract infection, site not specified",
    "R51.9": "Headache, unspecified",
    "Z00.00": (
        "Encounter for general adult medical examination without abnormal findings"
    ),
}


def bundled_icd10cm_concepts() -> list[Concept]:
    """The bundled ICD-10-CM subset as concepts (for ``ConceptNormalizer``).

    Example:
        >>> {c.code for c in bundled_icd10cm_concepts()} >= {"I10", "E11.9"}
        True
    """
    return [
        Concept(code=code, system=CodeSystem.ICD10CM, display=display)
        for code, display in _ICD10CM_SUBSET.items()
    ]


@TERMINOLOGY_REGISTRY.register("terminology.general.bundled_minimal")
class BundledMinimalBackend(BaseTerminologyService):
    """Offline ICD-10-CM lookup over a small, bundled, verified subset.

    Example:
        >>> backend = BundledMinimalBackend()
        >>> backend.validate("E11.9", CodeSystem.ICD10CM)
        True
        >>> backend.resolve("E11.9", CodeSystem.ICD10CM).display
        'Type 2 diabetes mellitus without complications'
    """

    def resolve(self, code: str, system: CodeSystem) -> Concept | None:
        if system is not CodeSystem.ICD10CM:
            return None
        display = _ICD10CM_SUBSET.get(code)
        if display is None:
            return None
        return Concept(code=code, system=system, display=display)

    def validate(self, code: str, system: CodeSystem) -> bool:
        return self.resolve(code, system) is not None

    def is_authoritative(self, system: CodeSystem) -> bool:  # noqa: ARG002
        """Never: this is a small subset, so a code missing from it (or from a
        system it does not cover at all) is merely unconfirmed, not invalid."""
        return False

    def map(
        self,
        code: str,  # noqa: ARG002 -- BaseTerminologyService interface, unused by design
        from_system: CodeSystem,  # noqa: ARG002
        to_system: CodeSystem,  # noqa: ARG002
    ) -> list[Concept]:
        """Always returns ``[]`` -- a single bundled vocabulary carries no
        crosswalk data between systems. A real, disclosed limitation, not
        a silent no-op standing in for unimplemented behaviour."""
        return []
