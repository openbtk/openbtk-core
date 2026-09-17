# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is `0.x`, breaking changes may land in a minor release and are
recorded here.

## [Unreleased]

**M5 — Providers & Retrieval (in progress)**.

### Added — M5 (task 5.5 — `sends_data_offsite` + `PolicyError`
enforcement end to end)
- `tests/integration/test_offsite_policy_pipeline.py`: closes the gap
  task 5.2's own roadmap note anticipated ("this can only be REALLY
  tested once a real offsite provider like OpenAI/Anthropic exists").
  `tests/security/test_offsite_policy_enforcement.py` already proved the
  enforcement *mechanism* (`Registry.create`/`create_from_config`) in
  isolation, against fake test doubles on a private registry instance
  (deliberately, so the embedding contract suite's real no-opt-out sweep
  never trips over a fake offsite provider planted in a real global
  registry). This proves the same enforcement fires when a real
  `Pipeline` names a real registered offsite provider
  (`llm.general.openai`, `embedding.general.openai`) — blocked with no
  policy opt-in, and past construction (onto the executor's own,
  already-disclosed "no llm/embedding-category dispatch yet" limit —
  confirmed to be a genuinely different failure message, not the same
  block reported twice) once `PolicyConfig(allow_offsite_providers=True)`
  is set. A negative control (`llm.general.huggingface_local`,
  `sends_data_offsite=False`) confirms the gate is targeted, not a
  blanket restriction. No network call happens in any case: a blocked
  construction never reaches the SDK client, and an allowed one fails
  for the dispatch reason before ever reaching one either.

### Added — M5 (task 5.4 — `embeddings/`: PubMedBERT, BioBERT,
ClinicalBERT, SapBERT, MedCPT, OpenAI)
- `openbtk.embeddings.huggingface.HuggingFaceEmbeddingProvider`
  (`embedding.general.huggingface`) — one generic local `transformers`
  encoder-embedding implementation, parameterised by model/revision/
  pooling, serving every current biomedical BERT-family embedding model
  (see presets below) rather than a bespoke class per model. `dimension`
  is a *required* constructor argument (not derived by loading the
  model): this class's own `sends_data_offsite`/`dimension`/
  `model_identity`/`provenance` contract checks must stay possible
  without a real model load. `embed()` still verifies the declared
  dimension against what the model actually produces on first real use,
  raising `ProviderError` on a mismatch rather than silently returning
  the wrong shape into a vector store.
- `openbtk.embeddings.openai.OpenAIEmbeddingProvider`
  (`embedding.general.openai`) — thin adapter over the `openai` SDK's
  Embeddings API, same retry/error-translation shape as
  `llms.openai.OpenAIProvider`. `dimension` defaults from a small table
  of OpenAI's own documented model widths, with a clear `ConfigError`
  for an unrecognised model unless `dimension=` is passed explicitly.
- `openbtk.embeddings.presets`: `BIOMEDICAL_EMBEDDING_PRESETS`,
  `list_embedding_presets()`, `create_embedding_preset()` — same "config,
  not classes" shape as `llms.presets` (task 5.3), covering PubMedBERT,
  BioBERT, ClinicalBERT, SapBERT and MedCPT. Every `model`/`revision`/
  `dimension` verified directly against the HuggingFace Hub API and each
  model's own `config.json`, not fabricated. Two real findings from that
  verification: **PubMedBERT was renamed** on the Hub (the well-known
  name 307-redirects to `microsoft/BiomedNLP-BiomedBERT-base-uncased-
  abstract-fulltext`; the preset keeps the familiar `"pubmedbert"` name
  but points at the real, current repository) and **MedCPT is a dual
  encoder**, not one symmetric model (NCBI publishes a separate
  Query-Encoder and Article-Encoder) — presented as two explicit presets,
  `medcpt-query` and `medcpt-article`, rather than picking one and
  calling it "medcpt". Pooling defaults to `"mean"` for the three plain
  MLM checkpoints (PubMedBERT/BioBERT/ClinicalBERT) and `"cls"` for
  SapBERT/MedCPT, per their own documented convention.
- `openbtk.core.retry`: `retry_with_backoff` moved here from
  `openbtk.llms.base` (re-exported there for backward compatibility)
  once `embeddings.openai` needed the identical rate-limited-then-retry
  shape and had nothing LLM-specific to justify importing it from an
  unrelated, same-level provider category.
- `tests/unit/embeddings/test_{huggingface,openai,presets}.py` — 48
  tests, 100% coverage across the whole `embeddings` package, none
  needing the real `torch`/`transformers`/`openai` packages installed
  (verified in a genuinely clean zero-extras venv): the HuggingFace
  provider's tests use a small numpy-backed fake tensor with just enough
  surface for its real pooling arithmetic to run unmodified.
- `tests/contract/test_embedding_contract.py`, extended the same way
  `test_llm_contract.py` was in task 5.2: the four checks that never
  touch the network or a model run unconditionally (via
  `PolicyConfig(allow_offsite_providers=True)`, since OpenAI is offsite
  by design); `embed`/`embed_one` are skipped per key unless
  `OPENBTK_SLOW_TESTS=1` **and** the specific resource each needs is
  genuinely available (`OPENAI_API_KEY`, or `torch`+`transformers`
  actually importable) — using a genuinely tiny, real test-only BERT
  checkpoint (`hf-internal-testing/tiny-random-bert`, 126K parameters)
  rather than a real biomedical preset's multi-GB model, so an opt-in run
  of this suite never downloads gigabytes just to check a shape.

### Added — M5 (task 5.3 — biomedical LLM presets)
- `openbtk.llms.presets`: `BIOMEDICAL_LLM_PRESETS`, `list_llm_presets()`,
  `create_llm_preset()`. Explicitly "config, not classes"
  (docs/10_ROADMAP.md's own wording): MedGemma, Meditron and OpenBioLLM
  are ordinary causal LMs `HuggingFaceLocalProvider` already handles, so
  a preset is a `{"type": ..., "params": {...}}` dict — the same shape
  `Registry.create_from_config` already accepts — not a new provider
  subclass per model. Every preset's `revision` is a real commit SHA
  fetched directly from the HuggingFace Hub API at write time (`GET
  /api/models/<id>`), not fabricated; `google/medgemma-4b-it` was
  checked and rejected for the MedGemma preset specifically because the
  Hub API reports it as `image-text-to-text` (multimodal), architecturally
  incompatible with `HuggingFaceLocalProvider`'s text-only interface —
  `google/medgemma-27b-text-it` (confirmed `text-generation`) is the
  MedGemma preset instead. `meditron-7b` and `medgemma-27b-text` are
  gated on the Hub (confirmed via the same API call) and require prior
  license acceptance + `huggingface-cli login`; disclosed in the module
  docstring, not silently assumed to work.

### Added — M5 (task 5.1 — `llms/base.py`: messages, responses,
retry/backoff, token accounting)
- `TokenUsage` moved from `core.provenance` to `core.schemas` (re-exported
  from `core.provenance` for backward compatibility with the name it
  shipped under in `v0.1.1`) so `LLMResponse` — also in `core.schemas` —
  can carry a `usage: TokenUsage | None` field without a real import cycle
  (`core.provenance` already imports `JsonValue` from `core.schemas`).
  `TokenUsage.__add__` sums a run's multiple LLM calls into one running
  total for `RunManifest.token_usage`; the executor still has no
  `llm`-category step to populate that field automatically.
- `tests/unit/core/test_schemas.py`: new file — `core.schemas` had no
  dedicated unit tests anywhere before this (only contract-suite and
  doctest coverage). 100% coverage on `core/schemas.py`.
- `src/openbtk/llms/base.py`: `retry_with_backoff()`, exponential backoff
  with jitter on `RateLimitError`, an injectable `sleep` so tests run
  instantly; re-exports `Message`/`LLMResponse`/`TokenUsage`/
  `BaseLLMProvider` for a single ergonomic import ahead of task 5.2's
  concrete providers. 100% coverage.

### Added — M5 (task 5.2 — OpenAI, Anthropic, HuggingFace-local,
OpenAI-compatible endpoint providers)
- `openbtk.llms.openai.OpenAIProvider` (`llm.general.openai`) — a thin
  adapter over the official `openai` SDK: Chat Completions for
  `generate`/`chat`, native SSE for `stream`, `TokenUsage` from the
  response's own usage block, SDK exception translation
  (`RateLimitError`/`AuthenticationError`/`APIError` →
  `openbtk.core.errors`'s hierarchy) wrapped in
  `retry_with_backoff`. `sends_data_offsite = True`.
- `openbtk.llms.anthropic.AnthropicProvider` (`llm.general.anthropic`) —
  same shape over the official `anthropic` SDK, with two genuine API
  differences handled rather than papered over: a leading `role: "system"`
  `Message` is split into the Messages API's own top-level `system`
  parameter, and `max_tokens` (required by that API, unlike OpenAI's) gets
  a class default so callers don't have to name one every time. Streaming
  retries only the connection-open step (`__enter__`), never a
  partially-consumed stream, and always closes the context manager via
  `finally`. `sends_data_offsite = True`.
- `openbtk.llms.openai_compatible.OpenAICompatibleProvider`
  (`llm.general.openai_compatible`) — for self-hosted/third-party
  endpoints that speak the OpenAI wire format without being OpenAI (vLLM,
  Ollama, LM Studio, ...). Built directly on `httpx` (already a core
  dependency, and its first real caller in `src/`) rather than the
  `openai` package, since the wire format — not OpenAI's service — is the
  actual contract. `sends_data_offsite` is conservatively `True` at the
  class level always: `Registry.create`'s policy gate checks it before
  construction, before `base_url` is even known, so it cannot vary
  per-instance by how that URL resolves.
- `openbtk.llms.huggingface.HuggingFaceLocalProvider`
  (`llm.general.huggingface_local`) — local `transformers` text
  generation. `sends_data_offsite = False`, the only one of the four.
  `revision` is a *required* constructor argument (no default): FR-P-05
  requires an immutable pinned revision, and unlike an API-based provider
  (where the model name itself is the vendor's pinned unit), a bare HF
  model name with no revision resolves to whatever the hub's default
  branch is at load time. `chat()` uses the tokenizer's own chat template
  when the model ships one, falling back to a disclosed, plain
  role-prefixed transcript otherwise. `stream()` uses the standard
  `transformers` background-thread `TextIteratorStreamer` pattern — with
  a real fix found while testing it: `streamer.end()` must be called
  unconditionally in the generation thread's `finally`, not only on
  success, or a `generate()` failure before producing any token would
  leave the consuming generator blocked forever (`TextIteratorStreamer`
  only signals completion through its own on-success hook).
- `tests/unit/llms/test_{openai,anthropic,huggingface,openai_compatible}.py`
  — 81 tests total, 100% coverage on all four provider modules, none
  needing the real `openai`/`anthropic`/`torch`/`transformers` packages
  installed (verified directly in a genuinely clean zero-extras venv):
  OpenAI/Anthropic/HuggingFace mock the SDK/model surface directly (the
  same rationale as `llms/base.py`'s own tests);
  `OpenAICompatibleProvider`'s use a real `httpx.MockTransport` instead,
  since `httpx` is core and always installed.
- `tests/contract/test_llm_contract.py`, extended: the three checks that
  never touch the network or a model (`sends_data_offsite`,
  `model_identity`, `provenance`) now run unconditionally for all four
  providers (via `PolicyConfig(allow_offsite_providers=True)`, since three
  of the four are offsite by design and `Registry.create` refuses them
  otherwise). The three that make a real call are skipped per key unless
  `OPENBTK_SLOW_TESTS=1` **and** the specific resource each one needs is
  genuinely available: `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` for the two
  cloud providers, `torch`+`transformers` actually importable for
  `HuggingFaceLocalProvider` (found by direct reproduction —
  `OPENBTK_SLOW_TESTS=1` alone let this suite attempt, and fail, a real
  model load in this project's own dev venv, which has `transformers` but
  not `torch`), and a real endpoint URL via
  `OPENBTK_TEST_OPENAI_COMPATIBLE_BASE_URL` for the compatible provider
  (there is no such server this repo controls or can assume exists
  anywhere).
- `pyproject.toml`: `torch>=2.0` added to the `llms` extra, needed by
  `HuggingFaceLocalProvider` — `transformers` (already in `text`) needs a
  real tensor backend to run a model, and does not pull one in itself.

### Fixed — M5
- `.pre-commit-config.yaml`: mypy hook pin (`v1.11.2`) predates a behaviour
  change in how mypy narrows `isinstance` checks against a dunder method's
  same-typed `other` parameter, producing a false-positive "unreachable"
  error on `TokenUsage.__add__`'s `return NotImplemented` branch that the
  project's actual (unpinned, `mypy>=1.10`) toolchain does not raise.
  Reproduced directly against an isolated venv pinned to the hook's exact
  old version to confirm before bumping to `v2.3.1`, matching the same
  "stale local pin, live toolchain is newer" issue already fixed once this
  project for ruff and numpy. Also missing `httpx` from the same hook's
  `additional_dependencies` (task 5.2's `openai_compatible.py` is this
  hook's first file to import a core dependency beyond the four already
  listed) — found the same way, by actually running the hook rather than
  assuming a core dependency would just be there.

## [0.1.1] — 2026-09-17

**M1 — Core framework**, **M2 — De-identification**, **M3 — Clinical Text
+ Pipelines**, and **M4 — v0.1 Release**.

**Note on the version number:** `0.1.0` is skipped. A `v0.1.0` tag existed
briefly during release troubleshooting and, at some point, its wheel was
uploaded to real PyPI and then deleted; PyPI permanently refuses to accept
a re-upload of a previously-deleted filename (a deliberate anti-tampering
policy, not a bug), so `0.1.0` can never be published under this project
name again. `0.1.1` is this project's actual first real release.

**M3 exit criteria met**: the four-stage pipeline (load → deid → segment →
chunk) runs end-to-end on synthetic data, emits a `RunManifest`, and the
memory benchmark (task 3.9) passes — 0.056 GB peak RSS for 10M notes against
a 4 GB target. Task 3.10 (entity linking + ConText) is deferred to M5, per
its own documented P1/optional status in the roadmap, not newly descoped.

### Added — M4 (packaging and contribution docs, task 4.3/4.4)
- `CONTRIBUTING.md` — real dev-environment setup, the exact lint/type/test
  commands CI runs, coding conventions, and the PR process.
- `CODE_OF_CONDUCT.md` — Contributor Covenant v2.1, with a GitHub-based
  enforcement path (private security advisory or direct maintainer
  contact) rather than inventing a conduct-reporting email that does not
  exist yet.
- `README.md` rewritten to describe only what has actually shipped
  (M1–M3), replacing the pre-implementation placeholder — the exact
  "README discipline" the roadmap itself calls out, after v1 shipped a
  README advertising features and a quick-start that didn't work.
  Includes a real quick-start (load → de-identify → segment → chunk
  through the real `Pipeline` API) and the checked-in de-id F1 and memory
  benchmark numbers, not projections.
- `tests/unit/test_readme.py`: extracts and actually executes every
  ` ```python ` block in `README.md` in CI, so a quick-start going stale
  is a test failure, not a silent drift — the same guarantee
  `--doctest-modules` gives every docstring `Example`, applied to the
  README for the first time.

### Added — M4 (docs site, task 4.5)
- `mkdocs.yml` + `mkdocs/` — a real, build-verified MkDocs Material site
  (home, quick start, and an API reference generated from the actual
  source docstrings via `mkdocstrings`, not hand-duplicated). Lives in
  `mkdocs/`, not `docs/` — this repository's own `docs/` is the
  deliberately git-excluded internal design-doc folder (confirmed via
  `git ls-files`), so a public site generated in CI could never read from
  it regardless of intent.
- `.github/workflows/docs.yml` — builds on every push to `main` that
  touches the site or the source, and deploys with `mike` under a `dev`
  version (not aliased to `latest`, since no tagged release exists yet).
  Both the build and the `mike deploy`/`set-default` invocations were
  exercised directly (a local, unpushed `mike deploy dev` really produces
  a `gh-pages` commit) before being written into the workflow, not
  assumed to work from reading `mike`'s docs alone.
- Disclosed, not silently worked around: `mkdocs build --strict` fails on
  two real but harmless `mkdocs_autorefs` false positives, where a
  doctest `Example`'s own OUTPUT line (a plain Python list-of-strings
  literal) is misread as an attempted cross-reference. The rendered pages
  are correct either way; `docs.yml` builds without `--strict` for this
  reason, recorded as a comment in `mkdocs.yml` itself.

### Fixed — M4
- `pyproject.toml`'s `[project.urls]` pointed at
  `github.com/openbtk/openbtk` — the actual repository is
  `openbtk/openbtk-core`. Found while writing accurate install
  instructions for the README, not assumed correct.
- `.pre-commit-config.yaml`'s `ruff`/`ruff-format` hooks were pinned to
  `v0.6.9`, which predates ruff's `TCH`→`TC` rule-selector rename —
  `pre-commit install` (exactly what `CONTRIBUTING.md` now tells every
  new contributor to run) failed outright on `pyproject.toml`'s own
  `"TC"` entry. Bumped to `v0.16.5`, matching what the project's own dev
  dependency actually resolves to.
- `.pre-commit-config.yaml`'s `mypy` hook's `additional_dependencies`
  listed a bare, unpinned `numpy` — pulling in numpy ≥2.5's PEP 695-syntax
  stubs, a hard mypy parse error under this hook's own `python_version =
  3.11`, the exact failure mode `pyproject.toml`'s own `numpy>=1.26,<2.5`
  constraint already exists to prevent for the main dependency list.
  Pinned the hook's copy to match.
- Several pre-existing `detect-secrets` false positives (a doctest's
  example commit SHA, test fixtures proving secret redaction and env-var
  interpolation work) had no `pragma: allowlist secret` marker —
  `detect-secrets` had never actually been run as part of this project's
  own verification before now. Marked, and refactored the two duplicated
  literals into named constants so one marker covers all uses.
  `pre-commit run --all-files` is clean end to end for the first time.

### Fixed — M3
- `MIMICNotesLoader`'s real tests (unit and contract suite) had no gate for
  pandas actually being installed, unlike every spaCy/medspaCy/transformers-
  dependent test elsewhere (all gated behind `OPENBTK_SLOW_TESTS`). CI's own
  `test-core` job installs with **zero optional extras** (NFR-10) across a
  9-combination OS/Python matrix — a condition never actually exercised
  during M3 development, since every venv used to build and verify it
  happened to have the `text` extra installed. Reproduced directly in a
  genuinely clean venv (`pip install -e ".[dev]"` only, matching CI's exact
  command): 10 tests failed with `MissingDependencyError`/collection
  errors. Fixed by skipping `TestMIMICNotesLoader`'s pandas-requiring tests
  (`@pytest.mark.skipif`) and the contract suite's
  `loader.clinical_text.mimic_notes` parametrization when pandas is not
  importable — the one test that verifies the *missing*-dependency error
  path itself (which mocks `require()` and never needs pandas genuinely
  absent) is unaffected, in its own un-gated class. Verified green in both
  a zero-extras venv (625 passed, up from 10 failed) and the full-extras
  venv (636 passed, unchanged) before this fix was considered done.
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

[Unreleased]: https://github.com/openbtk/openbtk-core/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/openbtk/openbtk-core/releases/tag/v0.1.1
[0.0.1]: https://github.com/openbtk/openbtk-core/releases/tag/v0.0.1
