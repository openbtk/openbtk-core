# EHR

`openbtk.data.ehr` loads FHIR R4, OMOP and HL7 v2 into one `PatientRecord` schema.
It must be imported explicitly (it registers `loader.ehr.fhir`, `loader.ehr.omop`
and `loader.ehr.hl7v2`). Reading needs the `ehr` extra: `pip install "openbtk[ehr]"`.
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

::: openbtk.data.ehr.hl7v2.HL7v2Loader

## Timelines and text

::: openbtk.data.ehr.temporal.TemporalNormalizer

::: openbtk.pipelines.timeline.PatientTimelineSerializer

## Cohorts

::: openbtk.data.ehr.cohort.CohortBuilder

::: openbtk.data.ehr.cohort.has_condition

::: openbtk.data.ehr.cohort.has_medication

::: openbtk.data.ehr.cohort.has_procedure

::: openbtk.data.ehr.cohort.age_between

::: openbtk.data.ehr.cohort.quasi_identifiers
