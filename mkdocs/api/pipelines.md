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
