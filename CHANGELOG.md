# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is `0.x`, breaking changes may land in a minor release and are
recorded here.

## [Unreleased]

**M1 — Core framework**, **M2 — De-identification**, and **M3 — Clinical Text
+ Pipelines**. Not yet released.

**M3 exit criteria met**: the four-stage pipeline (load → deid → segment →
chunk) runs end-to-end on synthetic data, emits a `RunManifest`, and the
memory benchmark (task 3.9) passes — 0.056 GB peak RSS for 10M notes against
a 4 GB target. Task 3.10 (entity linking + ConText) is deferred to M5, per
its own documented P1/optional status in the roadmap, not newly descoped.

### Fixed — M3
- `openbtk.deid.engine.DeidEngine(mode=...)` crashed outright
  (`AttributeError` in `_compute_config_hash`) when `mode` was passed as a
  plain string rather than a `DeidMode` enum member — exactly what every
  registry/config-driven construction supplies, since `StepConfig.params`
  is JSON-safe only. Worse than a crash: `Transform`'s `self._mode is
  DeidMode.REDACT`-style identity checks would have silently never matched
  a plain string, falling through to the wrong (`DATE_SHIFT`) branch, had
  the crash not caught it first. Found by actually constructing
  `preprocessor.general.deidentify` through the real pipeline executor
  (task 3.8), not assumed. Fixed by coercing `mode` via `DeidMode(mode)`
  at the top of `DeidEngine.__init__` — idempotent for a real enum member,
  and raises a clear `DeidError` for a genuinely invalid string.

### Added — M3 (clinical text + pipelines, tasks 3.1–3.7)
- `openbtk.core.provenance`: `RunManifest`, `StepProvenance`, `DataDigest`,
  `GuardrailOutcome`, `TokenUsage` (ADR-0005's remaining provenance
  primitives, deferred since M1 pending `PipelineConfig`). `RunManifest.config`
  is a serialised `dict` snapshot, not a `PipelineConfig` object — typing it
  that way would make `core.provenance` import `core.config`, which imports
  `core.registry`, which imports `core.base`, which imports
  `core.provenance` — a real layering cycle. `GuardrailOutcome` aggregates
  per (guardrail, attachment point) rather than one entry per record, to
  keep a manifest itself bounded at real corpus scale.
- `openbtk.pipelines.executor` — the streaming DAG executor
  (docs/03_ARCHITECTURE.md §7): topologically orders steps, streams records
  through them lazily (`Iterator` composition, no materialisation), and
  always emits a `RunManifest` — success or failure, there is no manifest-off
  switch. Scoped, disclosed rather than silently assumed: a single linear
  chain only (no fan-in, no fan-out — genuine branching would need
  `itertools.tee`-style broadcast with its own memory trade-offs, not built
  yet); only `loader`/`preprocessor`/`chunker`/`segmenter` steps are
  executable (no real `embedding`/`vectorstore` component exists yet to
  validate a dispatch path against). A step's `StepProvenance.status` is
  `"failed"` only for the step whose OWN transformation call raised, found
  by wrapping each step's own component call (not a shared generic
  reraise) so an upstream failure is never misattributed downstream.
  Redacts any `key|token|secret|password|credential`-shaped config value
  before embedding the config snapshot in the manifest
  (docs/06_SECURITY_COMPLIANCE.md §3.7).
- `openbtk.pipelines.pipeline.{Pipeline, Step}` — the public builder API
  (`Pipeline(...).add(Step(...)).guard(...).run()`), plus
  `Pipeline.from_yaml`/`from_config`. Both surfaces converge on the same
  `PipelineConfig` before the executor ever sees them.
- `tests/benchmark/test_memory.py` (task 3.9, NFR-01): a real, checked-in
  measurement, not a projection — 10,000,000 synthetic notes streamed
  through the real four-stage pipeline (JSONL load → deid → segment →
  chunk), peak RSS **0.056 GB**, against a 4 GB target. Nightly only
  (`@pytest.mark.benchmark`, skipped by default; `OPENBTK_RUN_BENCHMARKS=1`
  to opt in). Peak RSS measured with stdlib/`ctypes` only (no new
  dependency): `resource.getrusage` on Linux/macOS, `GetProcessMemoryInfo`
  on Windows, written as a single `sys.platform`-branched function so
  mypy's platform-narrowing type-checks each branch only on its own
  platform — verified directly against `--platform win32/linux/darwin`,
  since CI's `mypy --strict` runs on `ubuntu-latest` while this was
  authored and run on Windows.

### Added — M3 (task 3.8 — integration tests)
- `tests/integration/test_clinical_text_pipeline.py`: load → deid → segment
  → chunk, end-to-end, through the REAL executor and REAL `clinical_text`
  components (`PlainTextLoader`, `DeidPreprocessor`, `SectionSegmenter`,
  `SectionAwareChunker`) — not test doubles, unlike
  `tests/unit/pipelines/test_executor.py`'s own suite. Since
  `Pipeline.run()` returns only a `RunManifest`, never the processed data,
  a small test-local guardrail attached at `"after:chunk"` captures chunk
  text in-process for the test's own assertions — a legitimate use of the
  documented guardrail-attachment mechanism, not a bypass. Covers both
  `DeidMode.REDACT` and `DeidMode.SURROGATE` end to end, and a real
  guardrail `BLOCK` halting the real pipeline.
- `tests/security/test_phi_in_run_manifest.py`: the adversarial,
  release-blocker test docs/07_TEST_CHARTER.md §3.5 names directly
  (`test_no_phi_in_run_manifest`), deferred at M1 pending `RunManifest`
  and the executor — both now exist. Runs the full labelled synthetic PHI
  corpus through the real pipeline and asserts none of its planted
  identifiers appear anywhere in the serialised manifest.

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
- `openbtk.data.clinical_text.preprocessing.DeidPreprocessor`
  (`preprocessor.general.deidentify`) — a thin `BasePreprocessor` adapter
  wrapping `openbtk.deid.DeidEngine`: de-identifies `record.text`, updates
  `deid_status`, and preserves the full `DeidReport` (identifiers and
  counts only, never PHI values) under `record.metadata["deid_report"]`
  rather than discarding it, for a future pipeline stage to attach to a
  `RunManifest`. When `record.patient_ref` is absent, falls back to
  `record.record_id` for SURROGATE/DATE_SHIFT consistency — a disclosed
  narrowing to per-record consistency, not a silent one.
- `tests/contract/test_preprocessor_contract.py` generalized the same way
  as the loader and chunker suites, for the same reason: `DeidPreprocessor`
  genuinely reads `record.patient_ref`.

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
