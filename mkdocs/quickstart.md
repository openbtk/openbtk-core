# Quick start

## Installing

| What you get | Command |
|---|---|
| Released to PyPI (`0.1.1`: core, de-identification, clinical text) | `pip install openbtk` |
| Everything on `main` (EHR, providers, retrieval, guardrails, interop) | `pip install "openbtk[text,ehr,retrieval,llms,langchain] @ git+https://github.com/openbtk/openbtk-core.git"` |
| Core + clinical text from source | `pip install "openbtk[text] @ git+https://github.com/openbtk/openbtk-core.git"` |
| Core only (registry, config, provenance — no modality) | `pip install "openbtk @ git+https://github.com/openbtk/openbtk-core.git"` |

Optional extras: `text`, `ehr`, `retrieval`, `llms`, `langchain`, `langgraph`.
A component that needs one says which, by name, when you construct it.

The optional NER recognizer additionally needs a downloaded spaCy model:

```bash
python -m spacy download en_core_web_sm
```

## A real pipeline

This example is extracted from the project README and executed in CI on
every change (`tests/unit/test_readme.py`), so it cannot silently go stale.

```python
import contextlib
import io
import tempfile
from pathlib import Path

from openbtk.data import clinical_text  # noqa: F401 -- registers loaders/preprocessors/chunkers
from openbtk.pipelines import Pipeline, Step

with tempfile.TemporaryDirectory() as notes_dir:
    (Path(notes_dir) / "note1.txt").write_text(
        "Chief Complaint:\n"
        "Patient SSN: 123-45-6789 reports chest pain.\n"
        "Plan:\n"
        "Admit for observation.\n"
    )

    pipeline = (
        Pipeline("clinical-rag")
        .add(Step("load", "loader.clinical_text.plain_text", path=notes_dir))
        .add(Step("deid", "preprocessor.general.deidentify", mode="surrogate"))
        .add(Step("segment", "preprocessor.clinical_text.section_segment"))
        .add(Step("chunk", "chunker.clinical_text.section_aware", max_tokens=200))
    )
    with contextlib.redirect_stdout(io.StringIO()):  # structured JSON logs, not errors
        manifest = pipeline.run()

assert manifest.status == "success"
for step in manifest.steps:
    print(f"{step.step_id}: {step.records_out} record(s)")
# load: 1 record(s)
# deid: 1 record(s)
# segment: 1 record(s)
# chunk: 2 record(s)
```

Every submodule you use (here, `openbtk.data.clinical_text`) must be
imported explicitly — `import openbtk` alone stays deliberately light (no
modality module, no heavy dependency, under 500&nbsp;ms) so `pip install
openbtk` with zero extras still works.

## Inspecting the run

`pipeline.run()` returns a [`RunManifest`][openbtk.core.provenance.RunManifest]
— never the processed records themselves. It carries per-step throughput,
component identity, and (when a guardrail is attached) aggregated
pass/warn/block counts, all with no document content anywhere in it:

```python
print(manifest.status)  # "success" or "failed"
print(manifest.run_id)  # a fresh id per run
for step in manifest.steps:
    print(step.step_id, step.component.class_name, step.records_out)
```

See the [API reference](api/pipelines.md) for the full shape of
`RunManifest`, `StepProvenance`, and the `Pipeline`/`Step` builder.
