# Pipelines

`Pipeline`/`Step` is the public builder API; both it and `Pipeline.from_yaml`
converge on the same `PipelineConfig` before the streaming executor ever
sees them. See [Core → Configuration](core.md#configuration) for
`PipelineConfig` and [Core → Provenance](core.md#provenance) for the
`RunManifest` a run returns.

```python
from openbtk.pipelines import Pipeline, Step

pipeline = (
    Pipeline("clinical-rag")
    .add(Step("load", "loader.clinical_text.plain_text", path="./notes"))
    .add(Step("deid", "preprocessor.general.deidentify", mode="surrogate"))
    .add(Step("chunk", "chunker.clinical_text.section_aware", max_tokens=512))
)
manifest = pipeline.run()
```

::: openbtk.pipelines.pipeline.Pipeline

::: openbtk.pipelines.pipeline.Step

## Checkpoint and resume

`Pipeline.run(checkpoint_path=...)` resumes a long run instead of restarting it; see
the [CLI guide](../guides/cli.md#checkpoint-and-resume-long-runs) for what this does and
does not guarantee before relying on it.

::: openbtk.pipelines.checkpoint.Checkpoint

::: openbtk.pipelines.checkpoint.save_checkpoint

::: openbtk.pipelines.checkpoint.load_checkpoint

## Cross-modal joins

Attach a patient's structured events to their notes; see the
[EHR guide](../guides/ehr.md#joining-notes-to-structured-data).

::: openbtk.pipelines.join.join_notes_to_events

::: openbtk.pipelines.join.index_patients

::: openbtk.pipelines.join.NoteWithEvents

::: openbtk.pipelines.join.MatchedEvent
