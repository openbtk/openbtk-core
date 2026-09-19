# Clinical text

The clinical-text modality turns notes into de-identified, section-aware chunks
ready for embedding or retrieval. It needs no optional extra for the steps
below (`pip install openbtk`).

```python
import contextlib
import io
import pathlib
import tempfile

# Importing registers the loaders, preprocessors and chunkers. The registry
# logs each registration as JSON; a real application would leave that on stderr.
with contextlib.redirect_stdout(io.StringIO()):
    from openbtk.data.clinical_text.chunking import SectionAwareChunker
    from openbtk.data.clinical_text.loaders import PlainTextLoader
    from openbtk.data.clinical_text.preprocessing import (
        DeidPreprocessor,
        SectionSegmenter,
    )
    from openbtk.deid.schemas import DeidMode

notes = tempfile.TemporaryDirectory()
pathlib.Path(notes.name, "note1.txt").write_text(
    "Chief Complaint:\n"
    "Chest pain for two days.\n"
    "Assessment:\n"
    "Likely musculoskeletal. Patient phone (555) 010-2345.\n"
    "Plan:\n"
    "Ibuprofen; follow up in one week.\n",
    encoding="utf-8",
)

with contextlib.redirect_stdout(io.StringIO()):
    (record,) = PlainTextLoader().load(notes.name)  # one record per .txt file
    record = DeidPreprocessor(mode=DeidMode.REDACT).process(record)
    record = SectionSegmenter().process(record)
    chunks = list(SectionAwareChunker(max_tokens=20).chunk(record))

assert "010-2345" not in record.text  # de-identified first
assert list(record.sections) == ["chief_complaint", "assessment", "plan"]
assert [c.section for c in chunks] == ["chief_complaint", "assessment", "plan"]
notes.cleanup()
```

## Why the steps run in this order

1. **Load** yields one `ClinicalTextRecord` at a time. Loaders stream; nothing
   reads a corpus into memory.
2. **De-identify** before anything that records offsets. Redaction changes the
   text's length, so a section span computed on the original would point at the
   wrong characters afterwards.
3. **Segment** finds section headers (`Chief Complaint:`, `Assessment:`, `Plan:` ...)
   and records each section's span on the record.
4. **Chunk.** `SectionAwareChunker` never splits across a section boundary, so
   an Assessment is not cut off from its Plan. A section longer than
   `max_tokens` is split inside itself.

## The same thing as a pipeline

The pipeline executor composes these steps from a YAML config, streams records
through them, and emits a `RunManifest` (per-step counts, component identity,
a digest of each input file; never note content):

```yaml
name: notes-to-chunks
steps:
  - id: load
    type: loader.clinical_text.plain_text
    params: {path: ./notes}
  - id: deid
    type: preprocessor.general.deidentify
    params: {mode: redact}
    after: [load]
  - id: segment
    type: preprocessor.clinical_text.section_segment
    after: [deid]
  - id: chunk
    type: chunker.clinical_text.section_aware
    params: {max_tokens: 200}
    after: [segment]
```

Check it, then run it, with the [CLI](cli.md): `openbtk validate pipeline.yaml`,
`openbtk run pipeline.yaml`.

## Limits worth knowing

* **Token counts are approximate by default.** `count_tokens_approximate` is a
  labelled heuristic. Pass a real tokenizer's counter as `count_tokens=` to the
  chunker when a model's context window is the constraint.
* **Names are not removed by default.** The default recognizer finds structured
  identifiers (SSN, MRN, phone, dates, ...). See the
  [de-identification guide](deidentification.md) for what to enable and what the
  [benchmarks](../benchmarks.md) do and do not show.
* `MIMICNotesLoader` reads MIMIC-shaped CSV you already hold under your own data
  use agreement; OpenBTK never downloads credentialed data.
