# EHR

`openbtk.data.ehr` loads FHIR R4 and OMOP into one `PatientRecord` schema.
It must be imported explicitly (it registers `loader.ehr.fhir` and
`loader.ehr.omop`). Reading needs the `ehr` extra: `pip install "openbtk[ehr]"`.
See the [EHR guide](../guides/ehr.md) for a worked example.

## Schemas

::: openbtk.data.ehr.schemas.PatientRecord

::: openbtk.data.ehr.schemas.Demographics

::: openbtk.data.ehr.schemas.Encounter

::: openbtk.data.ehr.schemas.CodedEvent

::: openbtk.data.ehr.schemas.Measurement

## Loaders

::: openbtk.data.ehr.fhir.FHIRLoader

::: openbtk.data.ehr.omop.OMOPLoader

## Timelines and text

::: openbtk.data.ehr.temporal.TemporalNormalizer

::: openbtk.pipelines.timeline.PatientTimelineSerializer

## Cohorts

::: openbtk.data.ehr.cohort.CohortBuilder

::: openbtk.data.ehr.cohort.has_condition

::: openbtk.data.ehr.cohort.has_medication

::: openbtk.data.ehr.cohort.has_procedure

::: openbtk.data.ehr.cohort.age_between
