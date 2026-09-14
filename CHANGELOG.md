# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is `0.x`, breaking changes may land in a minor release and are
recorded here.

## [Unreleased]

**M1 — Core framework**, **M2 — De-identification**, and **M3 — Clinical Text
(in progress)**. Not yet released.

### Added — M3 (clinical text, tasks 3.1–3.5)
- `openbtk.deid.schemas.DeidStatus` (`UNKNOWN`/`RAW`/`DEIDENTIFIED`/`SURROGATE`) —
  lives in `deid`, not `clinical_text`, because `ehr` needs it too and the
  layering rule keeps modalities independent of each other.
- `openbtk.data.clinical_text.schemas`: `ClinicalTextRecord` and
  `ClinicalTextChunk` (docs/05_DATA_MODALITY_SPEC.md §1.1). `sections` maps
  labels to `TextSpan`s into `text`, not copies of the text itself. A real
  `timestamp` validator rejects naive datetimes at the schema boundary.
- `openbtk.data.clinical_text.loaders`: `PlainTextLoader`, `JSONLLoader`,
  `MIMICNotesLoader` (streaming, chunked `pandas.read_csv`, file handle
  closed explicitly).
- `openbtk.data.clinical_text.preprocessing.SectionSegmenter` — two
  backends: a dependency-free rule-based header matcher, and a real
  `medspacy` `Sectionizer` wrapper (opt-in, `text` extra).
- `openbtk.data.clinical_text.tokenization`: `count_tokens_approximate`
  (whitespace, always available) and `count_tokens_exact` (a real
  HuggingFace tokenizer, opt-in, cached per model name).
- `openbtk.data.clinical_text.chunking`: `FixedTokenChunker` and
  `SectionAwareChunker` — the modality spec's own "part that matters",
  genuinely net-new logic. Sentence-then-word-boundary packing with the
  SAME token counter used for both the cut decision and the reported
  `token_count`, closing a v1 defect where the two could disagree. Verified
  against two Hypothesis properties from docs/07_TEST_CHARTER.md §3.4: no
  chunk exceeds `max_tokens`, and (for any text with at least one
  non-whitespace character) concatenated chunk texts reconstruct the
  source exactly. A chunk that would carry zero real tokens (a
  whitespace-only body) is omitted rather than fabricated to satisfy the
  schema's `token_count >= 1` constraint.
- `tests/contract/test_chunker_contract.py` generalized to a per-key record
  factory (`_make_record`), the same pattern already used for the loader
  contract suite: a modality chunker's real `RecordT` need not match the
  shared reference fixture's generic shape.

### Added — M2 (de-identification, flagship)
- `openbtk.deid.schemas`: `PHICategory` (the 18 HIPAA Safe Harbor
  identifier categories), `DeidMode`, `Detection`, `RiskEstimate`,
  `DeidReport`, `DeidResult`. No schema here can hold a matched PHI value —
  `Detection` omits the detected text by construction.
- `tests/fixtures/labelled_phi_corpus.py`: a deterministic, seeded,
  synthetic corpus with ground-truth PHI spans, covering 16 of 18
  categories (the other two, `FULL_FACE_PHOTO` and `BIOMETRIC_IDENTIFIER`,
  are not text-representable at all).
- `openbtk.deid.recognizers.base.BaseRecognizer` — the pluggable
  detection extension point — plus its own `RECOGNIZER_REGISTRY`, and
  `RuleRecognizer`, a regex-based recognizer covering the 14
  format-detectable categories (SSN, email, URL, IPv4, date, phone/fax,
  MRN, health plan ID, account number, license number, vehicle ID, device
  ID, and a generic unique-identifier pattern). Deliberately does not
  attempt `NAME` or `GEOGRAPHIC_SUBDIVISION` — no reliable regex shape
  exists for either; that is NER's job.
- `openbtk.deid.recognizers.ner.NERRecognizer` — spaCy-based NER (`en_core_web_sm`,
  chosen over a scispaCy biomedical model, which targets scientific
  entities rather than PERSON/GPE — see the module's own docstring) for
  `NAME` and `GEOGRAPHIC_SUBDIVISION`. Optional (`pip install openbtk[text]`
  + `python -m spacy download en_core_web_sm`), lazily imported — never
  loaded by `openbtk.deid`'s default `DeidEngine()` configuration, and
  requested explicitly via `recognizers=["rule", "ner"]`. A new
  `OPENBTK_SLOW_TESTS=1` gate (`tests/conftest.py`) skips its
  model-requiring tests by default, activating the `slow` marker
  `pyproject.toml` had declared but not yet enforced.
- `DeidEngine._shield_rule_detections_from_ner` — ADR-0006's own named
  mitigation ("high-precision rules run first and their spans are excluded
  from NER re-examination"), implemented after measuring that without it,
  spaCy's false-positive PERSON spans on structured "Label: VALUE" text
  (license numbers, URLs, even bare field labels) measurably regressed
  already-perfect rule-covered categories via `SpanMerger`'s "widest span
  wins" policy.
- `openbtk.deid.merger.SpanMerger` — the accuracy-bearing component:
  resolves overlapping detections (widest span wins, confidences combine
  by noisy-OR, ties break by recognizer priority).
- `openbtk.deid.consistency.ConsistencyStore` — stable, HMAC-keyed
  original-to-surrogate mapping that never stores the original value.
- `openbtk.deid.transforms.Transform` — applies `REDACT` / `TAG` / `HASH`
  / `SURROGATE` / `DATE_SHIFT` to detected spans. Date shifting is
  per-patient and interval-preserving, verified via a Hypothesis property
  test.
- `openbtk.deid.engine.DeidEngine` — the one-call public API
  (`DeidEngine(...).deidentify(text, patient_id=...)`), wiring
  recognizers → merge → recall-bias filtering → transform → report.
- `tests/accuracy/`: two checked-in, real-measured F1 baselines. The
  zero-extras default (`recognizers=("rule",)`): 14 of 16
  text-representable categories at a perfect 1.0 F1, zero false positives,
  overall F1 0.933. The full ensemble (`recognizers=["rule", "ner"]`,
  `@pytest.mark.slow`): `NAME` recall 0.0 → 0.96 (precision drops to ~0.38
  — a real, disclosed trade-off, not hidden); `GEOGRAPHIC_SUBDIVISION`
  recall 0.0 → ~0.08 (Faker street addresses are not a shape
  `en_core_web_sm` reliably recognizes); every rule-covered category
  unaffected (1.0, protected by the shield above); overall F1 0.922.
- Additional `tests/security/` coverage: `DeidReport` and de-identified-text
  leak tests against the labelled corpus, including a quantitative,
  slow-gated measurement of the full ensemble's real (partial) leak
  reduction.

**Remaining gap, tracked explicitly:** the opt-in `LLMVerifier` (task 2.9)
is not yet built.

### Added — M1 (core framework)
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
