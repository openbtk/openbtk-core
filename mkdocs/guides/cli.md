# Command line

`pip install openbtk` installs an `openbtk` command (also `python -m openbtk`).
The library's own log lines go to **stderr**; a command's result is the only
thing on **stdout**, so `--json` output can be piped straight into `jq` or a
script.

OpenBTK's library log lines are verbose by default (every registration is a
`debug` line). Quiet them with `OPENBTK_LOG_LEVEL=warning` in the environment, or
`openbtk.core.logging.set_log_level("warning")` in code.

| Command | Does |
|---|---|
| `openbtk list [category]` | list registered components, e.g. `openbtk list loader` |
| `openbtk validate CONFIG` | check a pipeline config without running it |
| `openbtk run CONFIG` | run a pipeline and write its manifest |
| `openbtk deid INPUT --out DIR` | de-identify a corpus without writing a config |
| `openbtk replay MANIFEST` | re-run a recorded manifest and compare |
| `openbtk doctor` | diagnose installed extras, models and credentials |

**Exit codes:** `0` success; `1` it ran and failed, or a check found a problem
(a pipeline failed, validation errors, a replay diverged); `2` the invocation was
wrong (bad path, bad config, unknown name).

## Explore

```bash
openbtk list                     # every category, with a count
openbtk list guardrail           # the keys in one category; [offsite] marks providers that send data out
openbtk list llm --json          # machine-readable
```

## Validate before you run

```bash
openbtk validate pipeline.yaml
```

`validate` instantiates nothing and reads no data. It checks that every step's
`type` is registered, that each parameter exists on the component's constructor
and required ones are present, that the `after` graph is acyclic (it may fan out and fan in), a step
names each predecessor once and a loader is a root, and that no step sends data off-site under a policy that forbids
it. It does not check that a chunker may follow the step before it (no such rule
is specified yet), nor that your paths exist.

## Run and audit

```bash
openbtk run pipeline.yaml                      # manifest -> the config's provenance.manifest_dir
openbtk run pipeline.yaml --manifest run.json  # or choose the path
```

A config with validation errors is refused before anything runs (exit `2`). A run
that fails still writes its manifest, with the failure recorded. Manifests hold
counts, component identity and input digests, never content, and API keys are
redacted from the config they snapshot.

## Fan-out and fan-in

A pipeline is a DAG, not only a chain. A step with **several dependents** feeds each of
them every record; a step with **several predecessors** reads their streams interleaved:

```yaml
name: two-views
steps:
  - id: load
    type: loader.clinical_text.plain_text
    params: {path: ./notes}
  - id: deid
    type: preprocessor.general.deidentify
    after: [load]
  - id: fixed
    type: chunker.clinical_text.fixed_token
    params: {max_tokens: 256}
    after: [deid]        # deid feeds two chunkers...
  - id: by_section
    type: chunker.clinical_text.section_aware
    params: {max_tokens: 256}
    after: [deid]        # ...each of which sees every de-identified note
```

Things worth knowing:

- **Fan-in is a merge, not a join.** `after: [a, b]` gives the step `a`'s and `b`'s records
  one at a time in turn. It does not pair records across streams: to attach a patient's
  structured events to a note, use `join_notes_to_events`. A step that cannot handle a
  record's type fails, naming itself.
- **A diamond delivers twice.** If `b` and `c` both read `a` and `d` reads `b` and `c`,
  `d` receives every record of `a` once by each path.
- **Memory stays bounded.** The run pulls one record from each end of the graph in turn,
  so the buffer between branches holds only the lag between them (a chunker that turns one
  note into many widens it by that many), never the corpus.
- A guardrail attached to a step sees each of its records **once**, before the split.
- Every step is in the manifest with its own counts; a failure names the branch it
  happened in. A step may name each predecessor only once, a loader must be a root, and
  cycles are still refused by `validate`.

## Replay a run

```bash
openbtk replay run.json
```

`replay` rebuilds the pipeline from the config recorded in the manifest, runs it
again, and compares the two: input content, each step's record counts, and the
final status. A single input file is compared by SHA-256. A directory has no
content hash, so for it only the record count is compared and the report says so
(a note, not a failure). Exit `0` means they match; exit `1` lists what changed,
which is how you find out that the data moved or that a step is not
deterministic. A manifest whose config had secrets redacted cannot be replayed on
its own (it says which fields); re-run from the original config with the secrets
supplied as `${ENV_VAR}`.

## De-identify without a config

```bash
openbtk deid notes/ --out clean/ --mode redact
openbtk deid notes.jsonl --out clean/ --recognizers rule,ner
```

Input is a directory of `.txt` notes or one `.jsonl` file. Output is the
de-identified text plus `deid_summary.json` (counts by category; never a detected
value). It refuses a non-empty `--out` unless you pass `--force`. Without
`--recognizers rule,ner` it warns that names and street addresses are **not**
detected; see the [de-identification guide](deidentification.md).

## Doctor

```bash
openbtk doctor
openbtk doctor --require text,ehr    # exit 1 unless those extras are ready (useful in CI)
```

It reports Python and OpenBTK versions, the six core dependencies, which optional
extras are complete, whether the spaCy model is present, and which conventional
credential variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN`,
`UMLS_API_KEY`) are set or not. **Credentials are reported by name only; a value
is never read into the report.** OpenBTK does not read these variables implicitly:
a config opts in with `${NAME}`, or a provider's own SDK does.
