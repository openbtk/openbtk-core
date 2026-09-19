# EHR

The EHR modality loads FHIR R4 and OMOP into a single `PatientRecord` schema, so
everything downstream (timelines, cohorts, guardrails) is source-agnostic.

```bash
pip install "openbtk[ehr]"        # fhir.resources, pyarrow, pandas, hl7apy
```

The example below builds a small FHIR R4 `Bundle` in the shape Synthea's
per-patient export produces (synthetic; no real patient data), then loads it,
anchors events to the encounter, selects a cohort and writes the timeline as
text.

```python
import contextlib
import io
import json
import pathlib
import tempfile

with contextlib.redirect_stdout(io.StringIO()):  # registry logs, not errors
    from openbtk.data.ehr.cohort import CohortBuilder, has_condition
    from openbtk.data.ehr.fhir import FHIRLoader
    from openbtk.data.ehr.temporal import TemporalNormalizer
    from openbtk.pipelines import PatientTimelineSerializer

bundle = {
    "resourceType": "Bundle",
    "type": "collection",
    "entry": [
        {"resource": {"resourceType": "Patient", "id": "p1", "gender": "female"}},
        {
            "resource": {
                "resourceType": "Encounter",
                "id": "enc-1",
                "status": "finished",
                "class": {"code": "IMP"},
                "subject": {"reference": "Patient/p1"},
                "period": {
                    "start": "2024-03-14T08:00:00Z",
                    "end": "2024-03-16T00:00:00Z",
                },
            }
        },
        {
            "resource": {
                "resourceType": "Condition",
                "id": "cond-1",
                "subject": {"reference": "Patient/p1"},
                "encounter": {"reference": "Encounter/enc-1"},
                "code": {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": "385093006",
                            "display": "Community-acquired pneumonia",
                        }
                    ]
                },
                "onsetDateTime": "2024-03-14T08:10:00Z",
            }
        },
        {
            "resource": {
                "resourceType": "Observation",
                "id": "obs-1",
                "status": "final",
                "subject": {"reference": "Patient/p1"},
                "encounter": {"reference": "Encounter/enc-1"},
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "6690-2",
                            "display": "WBC",
                        }
                    ]
                },
                "valueQuantity": {"value": 14.2, "unit": "10*3/uL"},
                "effectiveDateTime": "2024-03-14T08:20:00Z",
                "referenceRange": [{"low": {"value": 4.5}, "high": {"value": 11.0}}],
            }
        },
    ],
}

folder = tempfile.TemporaryDirectory()
pathlib.Path(folder.name, "p1.json").write_text(json.dumps(bundle), encoding="utf-8")

with contextlib.redirect_stdout(io.StringIO()):
    patients = [TemporalNormalizer().process(p) for p in FHIRLoader().load(folder.name)]
    pneumonia = list(CohortBuilder(patients).include(has_condition("385093006")))
    timeline = PatientTimelineSerializer().serialize(pneumonia[0])

assert [p.patient_id for p in pneumonia] == ["p1"]
assert "Community-acquired pneumonia (SNOMED 385093006)" in timeline.text
assert "WBC 14.2 10*3/uL (ref 4.5-11.0) [HIGH]" in timeline.text
folder.cleanup()
```

## What each step does

* **`FHIRLoader`** reads a directory of R4 `Bundle` JSON files (one patient per
  file) and validates each resource with `fhir.resources`. `OMOPLoader` reads
  the OMOP CDM v5.4 core tables (person, visit, condition, drug, measurement,
  procedure, observation) from **Parquet** files in `pyarrow` batches. Both yield
  `PatientRecord`s. OMOP's limits are real: Parquet only (no CSV or SQL), codes
  come from each table's `*_source_value` column (the concept vocabulary is not
  bundled, so you declare which code system a table's source values use), the
  `death` table is not read, and because OMOP does not group a patient's rows
  together, each event table is scanned in full before the first record is
  yielded, so memory scales with event count rather than batch size.
* **`TemporalNormalizer`** anchors events to their encounter so a timeline has a
  usable order even where a resource lacks its own timestamp.
* **`CohortBuilder`** composes `include` / `exclude` predicates
  (`has_condition`, `has_medication`, `has_procedure`, `age_between`, or your own
  function) over a stream. It never materialises the source.
* **`PatientTimelineSerializer`** turns a record into a normal
  `ClinicalTextRecord`. That is the seam between the two modalities: the output
  goes through the same de-identification, segmentation and chunking as any note.

## From EHR to retrieval-ready text

Because the timeline *is* a `ClinicalTextRecord`, de-identify it exactly like a
note before it goes anywhere else:

```python
import contextlib
import io

with contextlib.redirect_stdout(io.StringIO()):
    from openbtk.data.clinical_text.preprocessing import DeidPreprocessor
    from openbtk.deid.schemas import DeidMode

    clean = DeidPreprocessor(mode=DeidMode.REDACT).process(timeline)

assert clean.deid_status.value == "deidentified"
```

## Guardrails for EHR data

`guardrail.ehr.code_validity` checks that coded events use codes their system
recognises, `guardrail.ehr.referential` finds events that point at a missing
encounter or fall outside its window, and `guardrail.ehr.units` flags
implausible lab units and values. They return results rather than raising; see
the [guardrails reference](../api/guardrails.md).

## Limits worth knowing

* The FHIR loader is exercised on Synthea-*shaped* bundles and resource-by-resource
  against `fhir.resources` validation. It has not been run against a full real
  Synthea export or a real hospital's FHIR server.
* The code-validity check consults the bundled ICD-10-CM subset unless you supply
  a fuller terminology service; a code it cannot confirm is a warning, never
  silently valid.
