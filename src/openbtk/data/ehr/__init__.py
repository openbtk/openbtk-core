"""EHR / EMR: FHIR + OMOP loaders, temporal normalisation, cohort building.
See docs/05_DATA_MODALITY_SPEC.md section 2.

Submodules are imported here (registering every component as a side effect)
for the same reason ``openbtk.data.clinical_text`` does: none of them need
their optional dependency (``fhir.resources``, ``pyarrow``) merely to be
*defined* -- only ``require()`` inside ``.load()`` does, resolved lazily on
first real use (CLAUDE.md rule 5).
"""

from __future__ import annotations

from openbtk.data.ehr import fhir, omop

__all__ = ["fhir", "omop"]
