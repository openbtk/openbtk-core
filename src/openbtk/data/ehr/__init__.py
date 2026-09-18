"""EHR / EMR: FHIR + OMOP loaders, temporal normalisation, cohort building.
See docs/05_DATA_MODALITY_SPEC.md section 2.

``PatientTimelineSerializer`` (task 6.5) is NOT re-exported here -- it lives
in ``openbtk.pipelines`` instead, since it is a cross-modal converter that
needs both this package's and ``openbtk.data.clinical_text``'s concrete
types, which import-linter's "Modalities are independent of one another"
contract forbids this package itself from importing (see
``openbtk.pipelines.timeline``'s own module docstring).

Submodules are imported here (registering every component as a side effect)
for the same reason ``openbtk.data.clinical_text`` does: none of them need
their optional dependency (``fhir.resources``, ``pyarrow``) merely to be
*defined* -- only ``require()`` inside ``.load()`` does, resolved lazily on
first real use (CLAUDE.md rule 5).
"""

from __future__ import annotations

from openbtk.data.ehr import cohort, fhir, omop, temporal

__all__ = ["cohort", "fhir", "omop", "temporal"]
