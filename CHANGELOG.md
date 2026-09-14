# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is `0.x`, breaking changes may land in a minor release and are
recorded here.

## [Unreleased]

**M1 — Core framework.** Not yet released.

### Added
- Exception hierarchy (`openbtk.core.errors`): one root `OpenBTKError` with
  structured, PHI-free `.context`, plus specific subclasses per failure mode
  (`ConfigError`, `RegistryError`, `PolicyError`, `LoaderError`,
  `ProviderError`, `GuardrailViolation`, and others).
- Lazy optional-dependency loading (`openbtk.core._lazy.require`), naming the
  pip extra to install rather than a bare `ImportError`.
- Core Pydantic schemas (`openbtk.core.schemas`): `TextSpan`, `GuardrailResult`,
  `LinkedEntity`, `SourceRef`, `SearchResult`, `Concept`, `Message`,
  `LLMResponse`, all frozen and `extra="forbid"`.
- Run provenance primitives (`openbtk.core.provenance`): `ModelIdentity`
  (rejects floating tags like `"latest"`) and `ComponentProvenance`.
  `RunManifest` deferred to a later milestone.
- 12 abstract base classes (`openbtk.core.base`) covering every extension
  point: loaders, preprocessors, chunkers, segmenters, feature extractors,
  embedding and LLM providers, vector stores, guardrails, rerankers, dataset
  adapters, and terminology services.
- `Registry[T]` (`openbtk.core.registry`): config-driven component lookup by
  permanent string key (`<category>.<scope>.<name>`), with alias support and
  12 global category registries.
- PHI-safe structured logging (`openbtk.core.logging`): a deny-list redaction
  processor, hashed identifiers, and an independent per-logger processor
  chain that never calls `structlog.configure()` globally.
- Declarative pipeline configuration (`openbtk.core.config`): `PipelineConfig`
  with YAML loading, `${VAR}` environment interpolation, and registry/DAG
  validation with real cycle detection.
- Plugin discovery via entry points (`openbtk.core.plugins`), loaded lazily
  on first registry lookup.
- `sends_data_offsite` policy enforcement in `Registry.create` /
  `create_from_config`: constructing a component that sends data off-site
  raises `PolicyError` unless the caller explicitly passes a policy with
  `allow_offsite_providers=True` — enforced even when no policy is passed at
  all, so the safe default cannot be bypassed by omission.
- Shared contract test suite (`tests/contract/`) parametrized over every
  registered implementation for all 12 base classes, with reference
  implementations so the suite is never vacuous.
- Security test suite (`tests/security/`): adversarial PHI-shaped strings
  fuzzed through real error and logging paths, a fixture-hygiene scanner for
  committed test source, and regression tests for offsite-policy enforcement.
- 100% statement and branch coverage on `openbtk.core`.

## [0.0.1] — 2026-08-30

**Placeholder release. The package installs and imports; it does not yet do
anything.** Published to reserve the name on PyPI and to exercise the release
pipeline end to end on a version that does not matter. Do not build on this.

### Added
- `src/openbtk/` package skeleton: 16 subpackages, each documenting its intended
  contents, plus a PEP 561 `py.typed` marker.
- Packaging via hatchling with `hatch-vcs`, so the version is derived from the
  git tag and cannot drift from it.
- Exactly six core dependencies — `pydantic`, `numpy`, `structlog`, `pyyaml`,
  `httpx`, `typing-extensions` — with a test asserting the budget and rejecting
  `langchain`, `langgraph`, `torch`, `transformers` and `spacy` from core.
- Optional extras: `text`, `ehr`, `retrieval`, `llms`, `langchain`, `all`,
  `dev`, `docs`.
- 23 packaging tests: import with zero extras, import from a working directory
  outside the repository, wheel layout, `py.typed` presence, forbidden package
  names, and the core dependency budget.
- Three enforced import-linter contracts: layered architecture, modality
  independence, and LangChain confined to its optional adapter.
- Pre-commit hooks: ruff, `mypy --strict`, detect-secrets, plus project guards
  against the `opentbtk` typo, `src.` imports, and `load_all()` in examples.
- CI across 3 operating systems × 3 Python versions, running the suite from
  outside the repository so a path-relative import cannot fake a pass.
- Release workflow: tag-triggered, main-only, TestPyPI verification across the
  full matrix before PyPI, publishing via Trusted Publishing with attestations.
- Apache-2.0 `LICENSE`, bundled into the distribution.

### Notes
This release deliberately contains no functionality. The previous codebase was
removed rather than repaired: it was never importable — three package names
coexisted in one repository, and no test had ever been executed against an
installed dependency. It is preserved on the `legacy/v1-snapshot` branch.

[Unreleased]: https://github.com/openbtk/openbtk-core/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/openbtk/openbtk-core/releases/tag/v0.0.1
