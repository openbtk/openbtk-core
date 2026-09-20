# Guardrails

A guardrail always returns a `GuardrailResult`; it never raises for a failed
check. Whether a `BLOCK` halts a pipeline is the caller's policy. Import
`openbtk.guardrails` to register them all.

## General

::: openbtk.guardrails.phi_leakage.PHILeakageGuardrail

::: openbtk.guardrails.terminology_validity.TerminologyValidityGuardrail

::: openbtk.guardrails.groundedness.GroundednessGuardrail

::: openbtk.guardrails.groundedness.GroundednessCheckInput

::: openbtk.guardrails.dose.DosePlausibilityGuardrail

::: openbtk.guardrails.dose.DoseLimit

## EHR

::: openbtk.guardrails.ehr.EHRCodeValidityGuardrail

::: openbtk.guardrails.ehr.ReferentialIntegrityGuardrail

::: openbtk.guardrails.ehr.UnitPlausibilityGuardrail

## Composing guardrails

::: openbtk.guardrails.pipeline.GuardrailPipeline

::: openbtk.guardrails.pipeline.GuardrailPipelineResult
