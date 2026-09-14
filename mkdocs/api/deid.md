# De-identification

`openbtk.deid` is cross-modal by design and has zero heavy dependencies in
its default configuration (its built-in `RuleRecognizer` is stdlib `re`
only) — importing it eagerly is safe.

```python
from openbtk.deid import DeidEngine, DeidMode

engine = DeidEngine(mode=DeidMode.SURROGATE, recognizers=["rule"])
result = engine.deidentify(text, patient_id="hashed-123")
```

::: openbtk.deid.engine.DeidEngine

::: openbtk.deid.schemas.DeidMode

::: openbtk.deid.schemas.DeidReport

::: openbtk.deid.schemas.DeidResult

::: openbtk.deid.schemas.DeidStatus

::: openbtk.deid.schemas.Detection

::: openbtk.deid.schemas.PHICategory

::: openbtk.deid.schemas.RiskEstimate
