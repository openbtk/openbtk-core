# Core framework

The registry, base classes, configuration, and provenance primitives every
other part of OpenBTK builds on. `import openbtk.core` (or its submodules)
never requires an optional dependency.

## Registry

::: openbtk.core.registry.Registry

::: openbtk.core.registry.get_registry

## Base classes

::: openbtk.core.base.Component

::: openbtk.core.base.BaseLoader

::: openbtk.core.base.BasePreprocessor

::: openbtk.core.base.BaseChunker

::: openbtk.core.base.BaseGuardrail

## Configuration

::: openbtk.core.config.PipelineConfig

::: openbtk.core.config.StepConfig

::: openbtk.core.config.PolicyConfig

::: openbtk.core.config.GuardrailConfig

## Provenance

::: openbtk.core.provenance.RunManifest

::: openbtk.core.provenance.StepProvenance

::: openbtk.core.provenance.DataDigest

::: openbtk.core.provenance.GuardrailOutcome

::: openbtk.core.provenance.ComponentProvenance

::: openbtk.core.provenance.ModelIdentity

## Errors

::: openbtk.core.errors.OpenBTKError
