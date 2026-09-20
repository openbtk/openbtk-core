# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is `0.x`, breaking changes may land in a minor release and are
recorded here.

## [Unreleased]

**M11 — v1.0 readiness, in progress.** The API freeze, the deprecation mechanism and
the security review are done. The PRD's v1.0 gate also asks for "all P2 requirements";
an audit found twelve of them unbuilt, and they are being added below. The adopter
(three named production users) and independent-evaluation items need people outside
this repository and are not claimed.

### Added — P2 requirements (PRD section 7, "v1.0 — Adoption")
- **`DosePlausibilityGuardrail`** (`guardrail.general.dose_plausibility`, FR-G-05).
  Finds drug, dose, frequency and route statements in generated text and checks them
  against limits **you supply**; OpenBTK ships no dose limits, and every limit must name
  its source. Blocks a dose or computed daily total over a limit. Says plainly when
  nothing was checked. A pass never means a dose is safe.
- **k-anonymity** (FR-D-11): `openbtk.deid.kanonymity` measures it (counts-only report)
  and reaches it by generalising along inspectable ladders, then suppressing the rare
  remainder; `guardrail.ehr.k_anonymity` checks an EHR cohort;
  `data.ehr.cohort.quasi_identifiers` builds the table. It covers only the columns you
  name and says nothing about what a group shares.
- **`HL7v2Loader`** (`loader.ehr.hl7v2`, FR-E-03): folds HL7 v2 messages (PID, PV1, DG1,
  OBX, RXE/RXA, PR1) into `PatientRecord`s. Names, addresses and phone numbers are never
  read. Codes map only through HL7 table 0396 names checked against HL7's table; plain
  `I10` is WHO ICD-10 and is not treated as ICD-10-CM.
- **Cross-modal joins** (FR-E-08): `openbtk.pipelines.join_notes_to_events` attaches a
  patient's events to their notes by encounter, by a configurable time window, or
  either. Identifier joins only; an unmatched note is reported, never guessed.
- **`ConceptNormalizer`** (FR-M-06): fuzzy free-text term to concept matching over a
  vocabulary you supply ("CBC" reaches its LOINC code through your alias). It refuses a
  candidate that differs in a number, laterality, acuity or negation word, so a shorter
  name never resolves to a more specific code.
- **`CrossEncoderReranker`** (`reranker.general.cross_encoder`, FR-R-04): MedCPT's
  cross-encoder pinned to its verified commit; local, lazy, keeps unscorable results.
- **Hybrid retrieval** (FR-R-05): `BM25Index` and `reciprocal_rank_fusion`;
  `RAGPipeline(bm25=...)` fuses exact-term hits with the vector store's before reranking.
- **Cloud LLM providers** (FR-V-01): `llm.general.azure_openai`, `llm.general.bedrock`
  (Converse API) and `llm.general.vertex` (`google-genai`); extras `bedrock` and
  `vertex`. Each is tested against fake SDKs **and** against the real SDK libraries
  offline (Bedrock's request passes botocore's own validator; the Vertex request is
  built from real `google.genai` types; real error objects are translated), with a CI
  job that fails on a skip. **None has been run against a live AWS, Google Cloud or
  Azure account.** An argument a provider cannot express is refused, not dropped.
- **Model cards** (FR-P-06): `openbtk.eval.model_card.ModelCard` and `cards_from_run`.
  Identity comes from the component's provenance; every metric must come from an
  `EvalManifest` (there is no way to type a score in); sections that need judgement
  read "Not provided." unless you write them.
- New guides: guardrails, terminology, retrieval; the evaluation guide gains model cards.

### Security
Found by an internal review ([`mkdocs/security-review.md`](mkdocs/security-review.md)).
An internal review is not an independent audit, and the page says so.
- **S-1: the UMLS API key appeared in tracebacks** (0.5.0). UMLS takes the key as a
  query parameter and `httpx` puts the URL in its error text; the chained error now
  carries only the exception type and HTTP status. *If you used
  `terminology.general.umls` with 0.5.0 and its tracebacks were logged somewhere
  shared, rotate the key.*
- **S-2: the email pattern was quadratic** in the input length (each doubling of a
  run like `a.a.a.a...` roughly quadrupled the time), so a crafted document could
  stall a worker. It is now linear and still matches a whole long address; every
  release since 0.1.0rc1 was affected.
- **S-3: the exact tokenizer was fetched from a moving Hub branch.** The default is
  pinned to a commit; a model of your own loads unpinned only with a
  `tokenizer.unpinned` warning until you pass `revision=`.
- **S-4:** two more quadratic patterns, in `parse_choice` (model output) and the
  sentence splitter.
- **S-6:** every GitHub Action, including the PyPI publisher, is pinned to a commit
  SHA; Dependabot keeps them current.

### Fixed
- **Emails on multi-label domains were only partly redacted** (S-5):
  `user@sub.example.co.uk` left `.co.uk` in the output. **Behaviour change:** more
  of such an address is now redacted, never less.

### Added — M11
- **The public API is frozen and enforced.** Public API is what the API reference
  documents. `tests/api/` snapshots its signatures, pydantic fields, the top-level
  `__all__`, the CLI, the manifest schema versions and every registry key, so a
  removed or changed name fails CI. (CLAUDE.md rule 6 already claimed a key
  snapshot; none existed until now.) Added: `mkdocs/stability.md` (what is public,
  what is breaking, the deprecation rules), and the API reference now covers the
  registry, all 13 base classes, shared schemas, configuration, provenance, logging,
  retry, deprecation and every error.
- **Deprecation mechanism**: `openbtk.core.deprecation` (`deprecated`,
  `warn_deprecated`, `OpenBTKDeprecationWarning`) and
  `Registry.register_alias(old, new, since=, removal=)`. OpenBTK's own tests turn the
  warning into an error, so nothing inside the library may call a deprecated API.
- `count_tokens_exact(..., revision=)` (additive).
- **CI `security` job** (bandit medium+, detect-secrets, pip-audit on the core install
  and the light extras) and a weekly `audit.yml`; `tests/security/test_redos.py`.

## [0.6.0] — 2026-09-20

### Read this first

**The de-identification benchmark on i2b2/n2c2 still has not been run.** The
harness and adapter ship and are tested, but that corpus is released only under a
Data Use Agreement and was not available. Every de-identification number
published (F1 0.933 rule-only, 0.922 with NER) is on a **synthetic** corpus: a
regression gate, not evidence of real-world accuracy. The default `DeidEngine`
**does not detect names or street addresses**; enable the opt-in NER recognizer
for those. Providers are tested against mocked SDKs and models in CI; the UMLS
`map()` call is unverified against the live service; the groundedness check is a
word-overlap heuristic. See `mkdocs/benchmarks.md`.

**Behaviour changes since 0.5.0 that can change your output** (all fix defects; see
"Fixed" below):
- `TerminologyValidityGuardrail` / `EHRCodeValidityGuardrail` with the default
  bundled backend now return a **WARNING ("unverifiable")** for a code the bundled
  subset cannot confirm, where 0.5.0 returned a **BLOCK ("does not exist")**. A
  BLOCK now requires a backend that is authoritative for that system. If you
  relied on the old BLOCK, inject a complete vocabulary.
- `DeidMode.DATE_SHIFT` now redacts non-date identifiers found beside dates; in
  0.5.0 such a document raised `DeidError`.
- Run manifests now record token-count settings such as `max_tokens` instead of
  `[REDACTED]`; credentials (`api_key`, `access_token`, ...) are still redacted.
- `PipelineConfig.validate_registry()` now reports unknown or missing step
  parameters and off-site components under a local-only policy; previously these
  surfaced only when the run tried to construct the component.

### What is new since 0.5.0

The `openbtk` command line, clinical QA and groundedness evaluation with eval
manifests, four guides plus a CLI guide, a full API reference, and eight
tutorial notebooks that are executed in CI. The six-dependency core is unchanged;
everything else is an optional extra (a new `notebooks` extra runs the
tutorials).

**M10 — Eval, CLI, docs — complete.** Writing the tutorials against the real
code found three genuine defects, fixed below; none of them was visible to the
existing test suite.

### Added — M10
- **`openbtk` command line** (`list`, `validate`, `run`, `deid`, `replay`,
  `doctor`; also `python -m openbtk`). Built on `argparse`, not Typer: a CLI
  framework would be a seventh runtime dependency for a core held at six, and it
  works from a plain `pip install openbtk`. A command's result is the only thing
  on stdout (library logging goes to stderr), so `--json` is pure JSON. Exit
  codes: `0` ok, `1` it ran and failed or a check found a problem, `2` the
  invocation was wrong.
  - `validate` (and `PipelineConfig.validate_registry`) now checks each step's
    parameters against its constructor signature (unknown or missing) and the
    off-site policy, without instantiating anything; `Pipeline.validate()` adds
    the executor's single-linear-chain rule.
  - `replay` rebuilds a pipeline from a manifest's recorded config, re-runs it and
    reports divergence in input content (SHA-256), per-step counts and status. A
    directory input has no content hash, so only its record count is compared and
    the report says so. A manifest with redacted secrets is refused, not replayed
    with a placeholder.
  - `doctor` reports installed extras, the spaCy model, and credentials **by name
    only** (a value is never read into the report).
- **Clinical QA evaluation** (`openbtk.eval.qa`): readers for the MedQA and
  MedMCQA file formats (verified against the datasets' Hub cards; a hidden or
  ambiguous answer key is refused, never guessed), a conservative answer parser,
  `evaluate_qa` with a Wilson 95% interval, per-subject counts, token accounting,
  and an `EvalManifest`. No dataset and no benchmark score ships (rule 14).
- **Groundedness scoring** (`openbtk.eval.groundedness`): claim-level
  faithfulness, and `evaluate_detector` to score the checker itself against human
  labels. States plainly that the default checker is a word-overlap heuristic.
- **`EvalManifest`** (FR-X-06): which component, which data (SHA-256), when, and
  counts; never question, answer or note text.
- **Documentation**: four guides (clinical text, de-identification, EHR,
  evaluation) plus a command-line guide; API reference pages for EHR, providers,
  retrieval, guardrails, terminology, evaluation and the LangChain adapter; a
  tutorials page. Every guide code block is executed, every pipeline YAML block
  validated, every `openbtk ...` command parsed, every API directive resolved, by
  `tests/docsite`.
- **Eight tutorial notebooks** (`notebooks/`), offline and synthetic-only, executed
  top to bottom in fresh kernels by `tests/notebooks`. New optional `notebooks`
  extra (nbclient, nbformat, ipykernel).
- **CI**: `docs-build` (`mkdocs build --strict` on every branch), `test-docs` and
  `test-notebooks` (all extras installed; fail on any skip). The docs build is now
  strict; four doctests whose list-of-strings output tripped mkdocs autorefs were
  rewritten.
- `openbtk.core.logging.set_log_level()` / `OPENBTK_LOG_LEVEL`: quiet OpenBTK's own
  log lines. The default is unchanged (every line).
- `BaseTerminologyService.is_authoritative(system)` (default `True`).

### Fixed — M10 (found by writing the tutorials)
- **The default terminology guardrail blocked valid data.** The bundled backend is
  a small ICD-10-CM *subset*, yet `validate()` returning `False` for anything
  outside it made `TerminologyValidityGuardrail` (and so `EHRCodeValidityGuardrail`)
  report real SNOMED, LOINC and most ICD-10-CM codes as "do not exist" with
  severity **BLOCK**. A partial vocabulary can fail to confirm a code, not call it
  invalid: those codes are now a WARNING ("unverifiable"). A BLOCK requires an
  authoritative backend (a complete vocabulary).
- **`max_tokens` was redacted from run manifests** (the credential pattern matched
  "token"), erasing a chunker's size limit from the audit record and making any
  pipeline with a chunker impossible to replay. Token-count keys are now exempt;
  `access_token`, `hf_token` and the like are still redacted.
- **`DeidMode.DATE_SHIFT` failed the whole document** if it contained any non-date
  identifier (a phone number beside a date), because every detection was fed to the
  date parser. Dates are shifted; every other identifier is now redacted.

## [0.5.0] — 2026-09-19

### Read this first

**The de-identification benchmark on i2b2/n2c2 has not been run.** The harness
and dataset adapter ship and are tested, but that corpus is released only under
a Data Use Agreement and was not available. Every de-identification number in
this release (F1 0.933 rule-only, 0.922 with NER) is on a **synthetic** corpus:
a regression gate, not evidence of real-world accuracy. The default
`DeidEngine` detects structured identifiers (SSN, MRN, phone, dates, ...) but
**does not detect names or street addresses**; enable the opt-in NER recognizer
for those. This is the one open gate for a "credible" v0.5 and it is tracked, not
hidden. See `mkdocs/benchmarks.md`.

Also worth knowing before you rely on it:
- LLM and embedding providers are exercised in CI against **mocked** SDKs and
  models; calls against live services run only when you supply keys or models.
- The UMLS terminology `map()` call has not been verified against the live
  service. The groundedness guardrail is a word-overlap heuristic, not entailment.
- The i2b2/n2c2 XML reader was written from the published annotation scheme and
  has only been run on hand-built files.
- The 10M-note memory figure (0.074 GB peak RSS) is one Windows run on
  synthetic notes.

### What is new since 0.1.1

The first PyPI release carrying EHR loading, the provider and retrieval layer,
guardrails and terminology, the evaluation harness and the LangChain adapter:
M5 (LLM and embedding providers, FAISS/Chroma/Qdrant, concept reranking, RAG
with source provenance), M6 (FHIR R4 and OMOP loading, timelines, cohorts, the
EHR-to-text serializer), M7 (PHI-leakage, terminology, groundedness and EHR
guardrails; terminology service), M8 (de-identification and retrieval
evaluation, benchmark harness, LangChain/LangGraph adapter) and M9 (release
gates, nightly memory benchmark, documentation brought up to date, live docs
site). Package classifier moves from Pre-Alpha to Alpha. The six-dependency
core is unchanged; everything else is an optional extra.

**M9 — v0.5 release readiness — built; `v0.5.0` deliberately NOT tagged.** One
PRD gate is open (below), and PyPI version numbers are permanent.

### Added — M9
- `tests/release/test_v05_gates.py`: every checkable v0.5 gate from the PRD is a
  test (loaders registered, provider/store/guardrail/terminology counts, the
  scheduled benchmark workflow, docs nav and workflows). The i2b2/n2c2 gate is a
  **strict xfail**: publishing a real result makes it fail until the xfail is
  removed, so the gate can only be closed on purpose.
- CI `test-ehr`: installs `[ehr]`, runs the FHIR/OMOP/Synthea-shaped suites and
  **fails if any test skips**. `test-core` installs zero extras and therefore
  skips them by design, so before this the EHR gate was demonstrated only
  incidentally.
- `benchmark.yml`: the literal NFR-01 benchmark (10,000,000 synthetic notes
  through load → de-identify → segment → chunk, peak RSS against 4 GB) runs
  nightly and on demand, writing its result to the job summary. The data is
  synthetic; MIMIC is credentialed and never used in CI.

### Measured — M9
- 10,000,000 synthetic notes through the full four-stage pipeline: peak RSS
  **0.074 GB** against the 4 GB target (Windows, Python 3.12, one run on
  2026-09-19, 34 min while other work shared the machine). This replaces the
  0.056 GB recorded at M3 in README and docs; both are far under the target.
  Peak RSS does not track corpus size (a 200,000-note run peaked at 0.097 GB),
  so this is mostly interpreter and import baseline. The nightly workflow
  produces the Linux figure.

### Changed — M9
- README, docs index and quick start still said only M1–M3 existed and that PyPI
  held a `0.0.1` placeholder. They now describe M1–M8, the real PyPI `0.1.1`,
  the extras, the verification limits of each area (cloud SDKs mocked in CI;
  UMLS mapping unverified against the live service; groundedness is a
  heuristic), and the missing n2c2 number. A test guards against the stale
  wording returning.

### Release readiness checked (not released)
- sdist and wheel build; `twine check --strict` passes; the wheel installs in a
  fresh venv with six core dependencies; every module outside the LangChain
  adapter imports with zero extras; the adapter fails with an actionable
  message; all extras resolve together.

### Closed — MkDocs site live
- GitHub Pages was enabled on 2026-09-19 (source `gh-pages`); the site is served at
  <https://openbtk.org/openbtk-core/> (the organisation's custom domain), and
  `site_url` now says so, so canonical links no longer point at a redirect.
  Checked by request: the index, quick start, benchmarks and LangChain pages
  all return 200 and the benchmarks page carries the "Not run" n2c2 statement.

### Open gates for v0.5
- **Published i2b2/n2c2 de-identification benchmark** — harness built, corpus
  not available (DUA). Deferred by the maintainer.
- Note: the LLM count is four provider classes plus three presets; counted as
  classes only it is four.

**M8 — Benchmarks & Interop — complete**, with one honest gap: the n2c2/i2b2
de-identification benchmark harness is built and tested, but **no i2b2/n2c2
number is published because none has been produced** — that corpus is
released only under a Data Use Agreement and is not available to this
project's CI or development environments.

### Added — M8 (tasks 8.1–8.2 — de-identification benchmark)
- `openbtk.eval.deid`: one scorer (relaxed entity-level overlap; category-aware
  and binary "any PHI" views; micro-averaged overall) now shared by the M2
  accuracy gate and the credentialed benchmark, so a published number and a CI
  gate cannot disagree. Reports hold counts and category names only — never
  document text. The M2 accuracy baselines pass unchanged on the shared scorer.
- `openbtk.deid.labelled`: `LabelledSpan` / `LabelledDocument`, the shared
  ground-truth types (in `deid` so both the dataset adapter and the evaluator
  can use them without breaking layering).
- `N2C2DeidDataset` (`dataset.clinical_text.n2c2_deid`): reads a
  user-supplied i2b2/n2c2 2014 `<deIdi2b2>` XML directory. Never downloads;
  without a path it raises `DatasetError` naming the registration page.
  Refuses (never mis-scores) files whose tag offsets do not match the text, and
  DOCTYPE/ENTITY declarations. Annotation types with no Safe Harbor category
  (AGE, PROFESSION, organisations) are counted as unscored, not dropped. **The
  parser was written from the published scheme and has not been run against the
  real corpus** — only hand-built files in its shape.
- `python -m openbtk.eval.deid_benchmark --dataset n2c2 --path DIR`: one
  command, JSON and Markdown reports, exit 2 on refusal.
- `mkdocs/benchmarks.md`: the reproducible synthetic-corpus tables (rule
  recognizer overall F1 0.933; rule + NER 0.922, category-aware), each with its
  regeneration command, the matching rule, and an explicit statement that
  n2c2/i2b2 was not run. `tests/accuracy/test_benchmarks_doc.py` compares the
  published tables with a fresh run so they cannot drift. The synthetic corpus
  is a regression gate, not evidence of real-world accuracy.

### Added — M8 (task 8.3 — retrieval metrics)
- `openbtk.eval.retrieval`: `recall_at_k`, `reciprocal_rank`, `ndcg_at_k`
  (linear gain, first-occurrence de-duplication), `RetrievalQuery`,
  `evaluate_retrieval` (streaming, O(1) aggregation) and `retriever_from`
  (embedding + vector store → retrieve function). No retrieval number is
  published: there is no agreed corpus in the repository yet.

### Added — M8 (tasks 8.4–8.6 — LangChain / LangGraph interop)
- `openbtk.integrations.langchain` (`pip install "openbtk[langchain]"`), the
  only package that imports LangChain: `OpenBTKEmbeddings`,
  `OpenBTKChatModel`, `OpenBTKVectorStore`, `chunk_to_document` /
  `document_to_chunk` (lossless round trip; a token count is never estimated),
  `as_runnable` / `from_runnable`, and `as_langgraph_node`. Tested against the
  real `langchain-core` and `langgraph`, not mocks of them.
- New optional extra `langgraph` (also in `all`), used to test the node
  adapter against a real `StateGraph`; not a core dependency.
- `mkdocs/langchain.md`: a guide whose every code block is executed by a test.
- CI: `test-langchain` (adapter against the real libraries) and
  `test-no-langchain` (rest of the suite with them verifiably absent) — the two
  halves of ADR-0001; `typecheck` now installs the `langchain` extra.
  Feature branches (`feat/**`, `fix/**`) now run CI before merge.
- `tests/packaging/test_langchain_isolation.py`: imports every non-adapter
  module under a blocker that makes LangChain/LangGraph unimportable, and
  checks the adapter fails with an actionable message when they are absent.
- The import-linter LangChain-ban contract now also covers `openbtk.embeddings`,
  `llms`, `retrieval` and `eval`.

### Known limits (M8)
- `OpenBTKChatModel` does not stream token-by-token (the provider `stream()`
  takes a prompt, not messages). `OpenBTKVectorStore` exposes the store's own
  scores and does not support normalised relevance scores or delete-all.

### Fixed
- CI `test-core` (zero optional extras) failed on `main` after M6 and M7:
  tests importing `fhir.resources` / `pyarrow` are now gated with
  `pytest.importorskip`.

**M7 — Guardrails & Terminology — complete**.

### Added — M7 (tasks 7.2–7.6 — `openbtk.guardrails`)
- `PHILeakageGuardrail` (`guardrail.general.phi_leakage`): wraps the real
  `DeidEngine`'s detections, never its transformed text. Result messages,
  details and spans carry only category names, counts and offsets — never
  a detected value.
- `TerminologyValidityGuardrail` (`guardrail.general.terminology`):
  validates codes from a `(code, system)` tuple, a `CodedEvent`/
  `Measurement`, a list, or a whole `PatientRecord` against an injectable
  terminology service (default: the bundled ICD-10-CM subset). A code the
  backend cannot confirm is a WARNING — never silently valid, never
  conflated with an invalid code (BLOCK).
- `GroundednessGuardrail` (`guardrail.general.groundedness`): decomposes
  an answer into claims and returns unsupported spans. Defaults to a
  dependency-free sentence splitter plus significant-word overlap —
  disclosed as a heuristic, not entailment — overridable via injected
  callables.
- `GuardrailPipeline`: runs guardrails in sequence over one payload,
  aggregates, short-circuits on BLOCK (configurable), and reshapes results
  into `GuardrailOutcome` entries for manifest recording. Separate from the
  executor's per-step guardrail attachment; unregistered, since its
  configuration is other guardrails.
- EHR guardrails: `guardrail.ehr.code_validity` (a real subclass of the
  terminology guardrail — not `register_alias`, whose docstring scopes it
  to deprecation transitions), `guardrail.ehr.referential` (dangling
  encounter refs; events outside their encounter window), and
  `guardrail.ehr.units` (a bounded UCUM allowlist plus wide ranges for
  seven common LOINC labs, each code verified against the real NLM
  Clinical Table service — a data-quality check, not clinical decision
  support). k-anonymity (`guardrail.ehr.k_anonymity`), named in the spec
  but not in this task's roadmap entry, is not built.
- Integration test over a real `FHIRLoader`-loaded `PatientRecord` and the
  real `DeidEngine`/`PatientTimelineSerializer`, including a proof that the
  default code-validity backend's ICD-10-CM-only scope genuinely fires for
  real SNOMED/LOINC codes.

### Added — M7 (task 7.1 — `openbtk.terminology`)
- `UMLSRestBackend` (`terminology.general.umls`): the real UMLS UTS REST
  API. Auth and endpoint behaviour verified directly against the live
  service (a real, well-formed 401 naming the `apiKey` query parameter)
  and NLM's current documentation, which deprecates the older
  ticket-granting-ticket flow. `sends_data_offsite=True`, so construction
  is policy-gated. Never exercised against a live licensed account (an
  approved individual licence is required) — verified against realistic
  mocked responses, the same treatment as OpenAI/Anthropic.
- `LocalVocabBackend` (`terminology.general.local`): a simple, documented
  `code,system,display` CSV — not any official release format (UMLS RRF,
  LOINC's multi-table export).
- `BundledMinimalBackend` (`terminology.general.bundled_minimal`): 20
  ICD-10-CM codes spanning many chapters, each checked against the real,
  free NLM Clinical Table Search Service. ICD-10-CM only, deliberately (a
  U.S. government work); SNOMED CT (restricted) and LOINC (redistribution
  requires its own licence acceptance) are not bundled.
- `CachedTerminologyService`: a TTL, content-addressed disk cache wrapping
  any terminology service for offline operation once warmed. Unregistered:
  its configuration is another service instance.

**M6 — EHR — complete**.

### Added — M6 (task 6.7 — integration: Synthea-shaped FHIR round trip)
A FHIR R4 Bundle matching Synthea's real default per-patient export shape,
run through the real `FHIRLoader -> TemporalNormalizer ->
PatientTimelineSerializer` chain, proves three claims for real rather than
by construction: (1) those three components genuinely compose; (2) the
"cross-modal seam" claim (docs/05_DATA_MODALITY_SPEC.md section 2.3) —
a PHI-shaped value deliberately embedded in a coded event's display text
is actually redacted by the exact same real `clinical_text` pipeline
(`PlainTextLoader` -> `DeidPreprocessor` -> `SectionSegmenter` ->
`SectionAwareChunker`) M3's own integration test uses, unmodified; (3)
"switch from FHIR to OMOP is a config change" — the identical, unmodified
`has_condition(...)` cohort predicate matches a patient loaded from a FHIR
bundle and an independently-built OMOP Parquet dataset describing an
equivalent patient.

### Added — M6 (tasks 6.5, 6.6 — `PatientTimelineSerializer`, `CohortBuilder`)
`PatientTimelineSerializer` converts a `PatientRecord` into a
`ClinicalTextRecord`, so a structured chart flows into the same
de-identification/chunking/guardrail text pipeline clinical notes already
use. **Lives in `openbtk.pipelines`, not `openbtk.data.ehr`** — a real
architectural issue caught by running import-linter, not assumed: placing
it under `data.ehr` fails the "Modalities are independent of one another"
contract, since the conversion genuinely needs both `data.ehr`'s and
`data.clinical_text`'s concrete types. `openbtk.pipelines` sits above
`openbtk.data` and may depend on either, the same way `RAGPipeline` (task
5.8) already spans provider categories. Separately, it is also not a
registered `preprocessor.ehr.*` step: `BasePreprocessor.process()` is
deliberately same-type in and out, which a `PatientRecord ->
ClinicalTextRecord` conversion cannot be.

`CohortBuilder` composes `include()`/`exclude()` predicates
(`has_condition`/`has_medication`/`has_procedure`/`age_between`) over a
streaming `Iterable[PatientRecord]`, never materialising the source. Does
**not** implement the original spec example's `before_index=True`/
`.within(encounter_window(...))` — that needs an index-date definition the
spec never actually gave, and inventing one would have been an unverified
design decision (rule 14). A follow-up ADR is the right place to define it
once a real use case needs it.

### Added — M6 (task 6.4 — `TemporalNormalizer`)
Anchors an event's missing timestamp to its encounter's start time when
resolvable (a real gap in OMOP's `visit_occurrence`-only timestamping in
particular), and sorts every event list chronologically, undated items
last. Genuinely fits `BasePreprocessor`'s same-type contract, unlike
`PatientTimelineSerializer`, so it registers normally as
`preprocessor.ehr.temporal`.

### Added — M6 (tasks 6.2, 6.3 — `FHIRLoader`, `OMOPLoader`)
`FHIRLoader` reads one FHIR R4 Bundle JSON file per patient (Synthea's own
default per-patient export shape), wrapping `fhir.resources`' R4B models
for real validation. Recognises `Patient`/`Encounter`/`Condition`/
`MedicationRequest`/`MedicationStatement`/`Procedure`/`Observation`; an
unrecognised coding system (outside the six standard HL7 URIs this loader
knows) is skipped with a logged warning rather than failing the whole
patient. Race/ethnicity read from the real US Core extensions when
present.

`OMOPLoader` reads OMOP CDM v5.4 core tables as batched Parquet via
pyarrow. Two disclosed, real scope decisions: no vocabulary resolution
(reads `*_source_value` columns directly — resolving standard concept ids
needs the multi-gigabyte `concept` table this project does not bundle);
and memory is not O(batch) across the whole loader the way the
`clinical_text` loaders are, since OMOP's normalised multi-table schema
does not co-locate one patient's events across independently-scanned
tables — a real, disclosed trade-off, documented in the module's own
docstring, not a shortcut.

### Added — M6 (task 6.1 — `ehr/schemas.py`)
`PatientRecord`/`Demographics`/`Encounter`/`CodedEvent`/`Measurement` —
the convergence point FHIR and OMOP both map into. `CodeSystem`
(`openbtk.core.schemas`) reconciles a real documentation inconsistency
flagged since M5: the spec's original `CodedEvent.system` comment
(SNOMED/ICD10/ICD10CM/RXNORM/LOINC/CPT) disagreed with the glossary's
authoritative set (SNOMED/LOINC/ICD10CM/RXNORM/CPT/UCUM) — reconciled in
favour of the glossary. `Measurement` gains a `display` field beyond the
original spec listing, needed for `PatientTimelineSerializer` to render a
human-readable lab name rather than a bare code.

**M5 — Providers & Retrieval — complete**.

### Added — M5 (task 5.8 — integration: full RAG pipeline with
`SourceRef` provenance)
- `openbtk.core.schemas.RAGAnswer` — `text`, `sources: list[SourceRef]`,
  `usage: TokenUsage | None`. Carrying `sources` as its own field (not a
  bare `LLMResponse`) is the whole point: every generated answer is
  traceable back to the exact retrieved chunks it was grounded in, in the
  order given to the model.
- `openbtk.pipelines.rag.RAGPipeline` — embed the question, query a
  vector store, optionally rerank, then generate, with `SourceRef`s
  reconstructed for every chunk used. Deliberately **not** another
  `openbtk.pipelines.executor` step type: that streaming DAG executor has
  no `embedding`/`vectorstore`/`llm`-category dispatch yet (confirmed
  directly by task 5.5's own integration test), and teaching it one is a
  real, separate, larger undertaking than this task's scope — the ingest
  side (load/deid/segment/chunk) already runs through the real executor
  today; `RAGPipeline` covers the query side, which does not. Depends
  only on the abstract base classes
  (`BaseEmbeddingProvider`/`BaseVectorStore`/`BaseLLMProvider`/
  `BaseReranker`), so it works with any real implementation from tasks
  5.2/5.4/5.6/5.7 or a test double, via dependency injection.
  Documents, and depends on, a real indexing convention: each chunk's own
  text/record_id/chunk_id must be present in the metadata dict passed to
  `BaseVectorStore.upsert` at index time (configurable key names) — there
  is nowhere else for them to live.
- `tests/unit/pipelines/test_rag.py` — 13 tests, 100% coverage, against
  small real subclasses of the actual abstract bases (not bare mocks).
- `tests/integration/test_rag_pipeline.py` — the real end-to-end proof:
  load → deid → segment → chunk through the REAL executor (same
  capture-guardrail pattern as M3's own integration test), then real
  embedding (`HuggingFaceEmbeddingProvider` with the same genuinely tiny,
  real `hf-internal-testing/tiny-random-bert` checkpoint the embedding
  contract suite uses), a real `FAISSVectorStore`, and a real
  `ConceptOverlapReranker` -- only the LLM is a deterministic test double
  (no real API credentials in this environment; already covered
  separately in tests/unit/llms). Gated on `torch`+`transformers` and
  `faiss-cpu` actually being installed. A real bug was caught and fixed
  while building this test, not merely by inspection: the first version
  never populated each indexed chunk's own `"cuis"` metadata, so
  `ConceptOverlapReranker`'s overlap score was silently 0 for every
  result and ranking fell back entirely to `tiny-random-bert`'s
  untrained (effectively random) embedding distances — caught by
  actually running the test and getting the wrong note back, not
  assumed correct from the code alone.

### Added — M5 (task 5.7 — `ConceptOverlapReranker`)
- `openbtk.retrieval.reranker.ConceptOverlapReranker`
  (`reranker.general.concept_overlap`) — reorders search results by
  shared UMLS CUIs between the query and each result (FR-R-03). Concept
  *extraction* is deliberately not built here: `openbtk.terminology`
  (UMLS/SNOMED/LOINC/RxNorm) isn't implemented yet, and UMLS itself is a
  licensed, restricted vocabulary this project already commits to never
  bundling — building a real entity-linker now would mean fabricating one
  against no real vocabulary, a correctness risk worse than the reranker
  itself. Concept extraction is an injected `extract_concepts:
  Callable[[str], Iterable[str]]` dependency instead — "wrap, don't
  reinvent" applied to this project's own future terminology work.
  Each result's own CUIs are read from `SearchResult.metadata` (a
  configurable key, `"cuis"` by default) rather than re-extracted per
  result at rerank time, since a real indexing pipeline runs entity
  linking once, at index time — not on every query. Sorts by
  `(overlap_count, original_score)`, both descending, so a result set
  with no concept metadata anywhere degrades gracefully to the original
  score order rather than an arbitrary one.
- `tests/unit/retrieval/test_reranker.py` — 15 tests, 100% coverage, no
  optional dependency and no mocking needed (a trivial word-set extractor
  stands in for a real linker in every test).
- `tests/contract/test_reranker_contract.py`, extended with the
  per-key constructor kwargs pattern already used by the vector store and
  LLM/embedding contract suites, since `ConceptOverlapReranker` has no
  default `extract_concepts`.

### Added — M5 (task 5.6 — `retrieval/`: FAISS, Chroma, Qdrant)
- `openbtk.retrieval.faiss.FAISSVectorStore` (`vectorstore.general.faiss`)
  — a local, in-process FAISS index (`IndexIDMap` over `IndexFlatL2`/
  `IndexFlatIP`) with a real string-id-to-int64 mapping layered over it
  (FAISS itself has neither string ids, metadata, nor upsert), real
  `remove_ids`-based delete, and real `write_index`/`read_index`
  persistence plus a JSON sidecar for the id/metadata state FAISS itself
  doesn't persist. `query`'s `filter` is an exact-match-all post-filter,
  disclosed as approximate (it can return fewer than `top_k` matches even
  when more exist) since FAISS's base index has no metadata-aware search
  at all -- mitigated, not hidden, by over-fetching when a filter is set.
  `SearchResult.score` is always higher-is-better regardless of metric
  (L2 distances negated, inner product used as-is).
- `openbtk.retrieval.chroma.ChromaVectorStore`
  (`vectorstore.general.chroma`) — wraps Chroma's *embedded* client modes
  only (ephemeral in-memory, or `PersistentClient`) — never `HttpClient`,
  a deliberate, disclosed scope limit (a remote-server variant would need
  its own offsite-policy handling). Found while testing against the real
  client, not assumed: Chroma rejects an empty `{}` metadata dict outright
  ("Expected metadata to be a non-empty dict") but accepts `None` for "no
  metadata" — handled by converting empty dicts before every upsert.
- `openbtk.retrieval.qdrant.QdrantVectorStore`
  (`vectorstore.general.qdrant`) — same embedded-only scope limit as
  Chroma (`":memory:"` or a local `path=`, never a remote/Cloud
  `url=`/`host=`). Qdrant point ids must be an unsigned int or UUID, never
  an arbitrary string — every point's id is a `uuid.uuid5` deterministically
  derived from the caller's own string id (stable across calls, so
  `upsert` on an existing id genuinely overwrites the same point), with
  the original string round-tripped through the point's payload. Another
  real finding from testing against the live client: on-disk `path=`
  access is exclusive, not concurrent — a second store instance pointed
  at a still-open path raises a real `RuntimeError` from the client
  itself, confirmed directly and documented, not silently papered over.
- Both Chroma's and Qdrant's persistence models don't match
  `BaseVectorStore`'s explicit `persist(path)`/`load(path)` pair (an
  embedded/local client persists continuously once configured with a
  path, with no "save now" step) — both classes disclose this and leave
  the base class's own `NotImplementedError` defaults in place rather
  than forcing an awkward, misleading implementation onto a model that
  doesn't have one; persistence is configured via each constructor's own
  `path` argument instead.
- `pyproject.toml`: `qdrant-client>=1.9` added to the `retrieval` extra —
  the roadmap names FAISS, Chroma **and** Qdrant, but the extra never
  listed the package the third one needs.
- `tests/unit/retrieval/test_{faiss,chroma,qdrant}.py` — 56 tests, 100%
  coverage across the whole `retrieval` package, run against the REAL
  libraries (not mocked): unlike the LLM/embedding SDK clients, all three
  are purely local, fast, no-network operations, so there is no cost or
  real-call concern to mock away — gated only on the respective package
  actually being installed (the same pattern already used for
  pandas-dependent loader tests), verified clean in a genuinely fresh
  zero-extras venv.
- `tests/contract/test_vectorstore_contract.py`, extended with the same
  missing-dependency gating and a real `tmp_path` for the persistence
  check (the previous version's bare relative path would have littered a
  real file in the working directory every run, for any store that
  actually supports `persist()`, once one existed to trip over it).

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
- `tests/contract/test_llm_contract.py` and
  `tests/integration/test_offsite_policy_pipeline.py`: both used a wrong
  commit SHA for `sshleifer/tiny-gpt2`
  (`5f91d94ce9ff8f65e1c2b0e75c7cd54306e02710`, apparently a transcription
  error made back in task 5.2/5.5 before this project's later,
  now-consistent habit of verifying every model identifier directly
  against the HuggingFace Hub API) — never actually exercised until a
  full `OPENBTK_SLOW_TESTS=1` run with real `torch`/`transformers`
  installed hit `RevisionNotFoundError: 404 Client Error... Invalid rev
  id`. Corrected to the real SHA
  (`5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be`), verified directly via
  `GET https://huggingface.co/api/models/sshleifer/tiny-gpt2`. A full
  sweep of every other model-identifier SHA introduced across M5 (11
  total, in `llms/presets.py`, `embeddings/presets.py`, and the
  contract/integration test fixtures) was re-verified against the real
  Hub API the same way; all others were already correct.
- `src/openbtk/llms/huggingface.py`: `HuggingFaceLocalProvider` raised a
  real `ValueError` loading `sshleifer/tiny-gpt2` once the SHA above was
  fixed and a real download actually happened — that repo ships only
  legacy slow-tokenizer files (`vocab.json` + `merges.txt`, no
  `tokenizer.json`), and the installed `transformers` version no longer
  silently falls back to the slow tokenizer when fast-tokenizer
  conversion fails. `_load_tokenizer()` now retries with `use_fast=False`
  on that `ValueError`, a real, disclosed trade-off (correctness over the
  fast tokenizer's speed) for any local model that ships without a
  bundled fast-tokenizer file, not just this test fixture.
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

[Unreleased]: https://github.com/openbtk/openbtk-core/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/openbtk/openbtk-core/releases/tag/v0.6.0
[0.5.0]: https://github.com/openbtk/openbtk-core/releases/tag/v0.5.0
[0.1.1]: https://github.com/openbtk/openbtk-core/releases/tag/v0.1.1
[0.0.1]: https://github.com/openbtk/openbtk-core/releases/tag/v0.0.1
