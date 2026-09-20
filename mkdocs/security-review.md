# Security review

An **internal** review of OpenBTK's own code and build, done for the 1.0 milestone
(M11, task 11.2) by the maintainers' tooling and reading of the source. It is **not** an
independent audit or a penetration test, and this page does not claim to be one; see
[what this review did not do](#what-this-review-did-not-do). To report a
vulnerability, use [SECURITY.md](https://github.com/openbtk/openbtk-core/blob/main/SECURITY.md).

OpenBTK handles protected health information, so the questions asked were the ones
that matter for it: can PHI or a credential reach a log, an error, a manifest or a
cache; can untrusted input stall or crash a worker; and what does installing and
releasing the package trust.

## Findings

Six findings, all fixed on the branch this page ships on. Each has a test that fails
without the fix.

| ID | Finding | Impact | Present in | Fixed by |
|---|---|---|---|---|
| S-1 | The UMLS API key was printed in tracebacks | Credential disclosure | 0.5.0 | Chained error is now the exception type and HTTP status only |
| S-2 | Email pattern was quadratic in the input length | Availability | every release since 0.1.0rc1 | Match may only start where a run of local-part characters starts |
| S-3 | The exact tokenizer was downloaded from a moving Hub branch | Supply chain | every release since 0.1.0rc1 | Default model pinned to a commit; other models warn until pinned |
| S-4 | Two more quadratic patterns: the QA answer parser and the sentence splitter | Availability | splitter: every release since 0.1.0rc1; QA parser: 0.6.0 | Unambiguous pattern; backwards scan |
| S-5 | Emails on multi-label domains were only partly redacted | PHI left in output | every release since 0.1.0rc1 | Domain accepts dot-separated labels |
| S-6 | CI and release actions were pinned by mutable tag | Supply chain | all | Pinned to commit SHAs, updated by Dependabot |

### S-1: the UMLS API key in tracebacks

UMLS takes its key as a **query parameter**. `httpx` puts the full request URL in the
text of its errors, and the backend chained the original error (`raise ... from e`),
so any traceback that reached a log, a CI console or a crash reporter contained
`apiKey=...`. The raised `TerminologyError` itself never carried it, which is why the
existing "no PHI in errors" tests did not catch it.

The chained cause is now a fresh exception naming only the error type and HTTP
status. **If you used `terminology.general.umls` with 0.5.0 and its tracebacks were
logged anywhere shared, rotate the key.** The OpenAI-compatible provider sends its key
in a header and its errors carry the URL only; that path was checked and needs no
change.

### S-2 and S-4: quadratic patterns

A regular expression whose running time grows with the *square* of the input turns one
crafted document into a stalled worker. Found by timing every pattern that reads
document or model text against adversarial input (long runs of `a.`, `a-`, digits,
spaces, `@`, and so on) at doubling sizes.

- The **email** pattern started a fresh scan at every word boundary, so `a.a.a.a...`
  rescanned the rest of the run each time, so each doubling of the input roughly
  quadrupled the time (reproduce it by running the old pattern,
  `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}\b`, on `"a." * n`). Every
  rule-based de-identification call was exposed to it. The `SN-` and `MRN-` shapes
  triggered it too.
- The QA **answer parser** (`\s*(?:is|:)?\s*`) had two adjacent whitespace groups that
  a run of spaces could be split between in quadratic ways; it parses *model output*.
- The **sentence splitter** re-ran a search over a slice that kept growing across a
  run of abbreviations (`Dr. Dr. Dr. ...`), and rescanned a long word from every
  position.

A length cap would have fixed the email time but matched only the *tail* of a long
address and left its head un-redacted, which is worse than being slow. The fix
instead anchors where a match may start, so every address is still matched whole
(pinned by a test). `tests/security/test_redos.py` feeds each pattern adversarial
input at 100 kB and requires a result within 3 seconds, which linear code beats by two
orders of magnitude and the old code missed by a wide margin.

### S-3: an unpinned model download

`count_tokens_exact` loaded a Hugging Face tokenizer by name, which follows the
repository's default branch: whoever controls that repository could change what code
and files are loaded. The default model is now pinned to a specific commit (the
current `main`, recorded in the source), and a model you name yourself is loaded
unpinned only with a warning until you pass `revision=`.

### S-5: a leak the timing work turned up

While fixing S-2 a test for `user@sub.example.co.uk` showed the old pattern matched
only `user@sub.example`, leaving `.co.uk` in the de-identified text. A domain suffix
is a weak identifier, but it is what a de-identifier is for.

### S-6: build and release supply chain

Every third-party GitHub Action, including the one that publishes to PyPI, was
referenced by a tag an upstream owner can move. They are now pinned to commit SHAs
(the tag is kept in a comment), and Dependabot proposes updates as reviewable pull
requests. PyPI publishing already used a protected environment with manual approval
and trusted publishing (no long-lived token).

## Checked, no issue found

- **No PHI or credentials in logs, errors, manifests or reports.** Covered by the
  existing adversarial `tests/security/` suite; the review additionally read every
  place an error context is built. `RunManifest` and `EvalManifest` hold identifiers
  and counts only, and the executor redacts parameters whose names look secret.
- **Static analysis.** Bandit 1.9.4 over `src/` reported five things, and none is
  left unexplained: the unpinned download (S-3, fixed); the stdlib XML parser (B405,
  B314: the n2c2 reader refuses any `DOCTYPE` or `ENTITY` first, the input that makes it
  unsafe, so `defusedxml` would add a seventh dependency for nothing); the retry
  back-off's use of `random` (B311: jitter, not a secret); and a false positive on the
  word "token" in a constant's name (B105). Each suppression carries its reason on
  the line, and CI fails on any new medium-or-higher finding.
- **No dynamic code execution.** No `eval`, `exec`, `pickle`, `subprocess`, `os.system`
  or `shell=True` in the library; configuration is read with `yaml.safe_load`.
- **Known vulnerabilities in dependencies.** `pip-audit` over the zero-extras install
  (the six core dependencies) and the light extras (`ehr`, `langchain`, `langgraph`)
  found none at the time of review. Auditing a full development environment, with the
  heavy extras, did report advisories in two third-party packages, neither of which
  has a fixed release, and neither of which OpenBTK's use reaches:
    - **`chromadb`** (five advisories, all about the Chroma *server*: authorisation
      between tenants, and remote code through a server API that loads a model
      repository). OpenBTK uses only the in-process client (`Client` and
      `PersistentClient`), never `HttpClient`, never runs a Chroma server, and never
      passes `trust_remote_code`. If you run a Chroma server yourself, that exposure is
      yours; keep it off untrusted networks.
    - **`nltk`** (a model-path validation advisory). OpenBTK does not depend on it; it
      arrives through `medspacy_quickumls` when that optional package is installed.
- **No secrets in the repository.** `detect-secrets` over every tracked file.
- **Off-site policy.** A component that would send text off the machine is refused
  by config validation unless the policy allows it; this is tested.
- **Constructors do no I/O**, so `openbtk validate` and config checks cannot trigger a
  download or a network call.

## Kept honest by CI

These checks now run on every pull request (the `security` job) and, for dependencies,
weekly against `main`, because a vulnerability can be published without any commit
here:

- `bandit -ll` over `src/`
- `detect-secrets` over tracked files
- `pip-audit` on the core install and the light extras
- the ReDoS timing tests, in the normal test run

## What this review did not do

- **It was not independent.** The people who wrote the code reviewed it, with
  tooling. For a v1.0 that hospitals rely on, an external review is still worth having;
  this page is the material to hand to one, not a substitute.
- **No fuzzing campaign and no penetration test.** The timing probes were targeted, at
  the patterns that read text; other parsers (FHIR, HL7 v2, OMOP, JSON Lines) rely on
  their parsing libraries and standard-library readers, and were read but not fuzzed.
- **Third-party models and services are outside its scope.** Model weights (spaCy,
  Hugging Face), LLM providers and the UMLS service are the operator's trust
  decisions. OpenBTK's controls are the off-site policy, pinning where it loads a model
  itself, and never logging content.
- **The heavy extras are not audited in CI** (torch, transformers, spaCy, FAISS,
  Chroma and similar): their advisories change often, and the ones found in the
  review are listed above. Track them with Dependabot or `pip-audit` on your own lock
  file.
- **De-identification is not perfect.** A recognizer that misses an identifier is a
  privacy failure of a kind no scanner finds. Measured recall is on the
  [benchmarks page](benchmarks.md), and the i2b2/n2c2 number is still to be published.

## Repeating this review

```bash
bandit -r src -ll                       # static analysis
pip-audit                               # in an environment with the extras you use
pytest tests/security                   # PHI-leak and timing tests
```

When you add a regular expression that reads document text, add its adversarial input
to `tests/security/test_redos.py`.
