# Core framework

The registry, base classes, configuration, and provenance primitives every
other part of OpenBTK builds on. `import openbtk.core` (or its submodules)
never requires an optional dependency.

Everything on this page is under the [stability policy](../stability.md): it is
what you write custom components and plugins against.

## Registry

::: openbtk.core.registry.Registry

::: openbtk.core.registry.get_registry

::: openbtk.core.registry.list_categories

::: openbtk.core.registry.ComponentInfo

## Base classes

Every component role has one. Subclass it, register it under a key, and it is
usable from a config file, the CLI and a pipeline.

::: openbtk.core.base.Component

::: openbtk.core.base.BaseLoader

::: openbtk.core.base.BasePreprocessor

::: openbtk.core.base.BaseChunker

::: openbtk.core.base.BaseSegmenter

::: openbtk.core.base.BaseFeatureExtractor

::: openbtk.core.base.BaseEmbeddingProvider

::: openbtk.core.base.BaseLLMProvider

::: openbtk.core.base.BaseVectorStore

::: openbtk.core.base.BaseReranker

::: openbtk.core.base.BaseGuardrail

::: openbtk.core.base.BaseDatasetAdapter

::: openbtk.core.base.BaseTerminologyService

## Shared schemas

::: openbtk.core.schemas.TextSpan

::: openbtk.core.schemas.LinkedEntity

::: openbtk.core.schemas.SourceRef

::: openbtk.core.schemas.SearchResult

::: openbtk.core.schemas.GuardrailResult

::: openbtk.core.schemas.GuardrailSeverity

::: openbtk.core.schemas.Message

::: openbtk.core.schemas.LLMResponse

::: openbtk.core.schemas.TokenUsage

::: openbtk.core.schemas.RAGAnswer

::: openbtk.core.schemas.CodeSystem

::: openbtk.core.schemas.Concept

## Configuration

::: openbtk.core.config.PipelineConfig

::: openbtk.core.config.StepConfig

::: openbtk.core.config.PolicyConfig

::: openbtk.core.config.ProvenanceConfig

::: openbtk.core.config.GuardrailConfig

::: openbtk.core.config.ValidationIssue

## Provenance

::: openbtk.core.provenance.RunManifest

::: openbtk.core.provenance.StepProvenance

::: openbtk.core.provenance.DataDigest

::: openbtk.core.provenance.GuardrailOutcome

::: openbtk.core.provenance.ComponentProvenance

::: openbtk.core.provenance.ModelIdentity

## Logging

::: openbtk.core.logging.get_logger

::: openbtk.core.logging.configure_logging

::: openbtk.core.logging.set_log_level

## Retrying

::: openbtk.core.retry.retry_with_backoff

## Deprecation

::: openbtk.core.deprecation.deprecated

::: openbtk.core.deprecation.warn_deprecated

::: openbtk.core.deprecation.OpenBTKDeprecationWarning

## Errors

Every OpenBTK error derives from `OpenBTKError`, carries a `context` dict of
identifiers only (never content), and is safe to log.

::: openbtk.core.errors.OpenBTKError

::: openbtk.core.errors.ConfigError

::: openbtk.core.errors.RegistryError

::: openbtk.core.errors.MissingDependencyError

::: openbtk.core.errors.PolicyError

::: openbtk.core.errors.PluginError

::: openbtk.core.errors.LoaderError

::: openbtk.core.errors.ProcessingError

::: openbtk.core.errors.DatasetError

::: openbtk.core.errors.DeidError

::: openbtk.core.errors.ProviderError

::: openbtk.core.errors.RateLimitError

::: openbtk.core.errors.AuthenticationError

::: openbtk.core.errors.RetrievalError

::: openbtk.core.errors.TerminologyError

::: openbtk.core.errors.GuardrailViolation
