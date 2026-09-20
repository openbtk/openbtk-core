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

## Structured data: k-anonymity

::: openbtk.deid.kanonymity.k_anonymity_report

::: openbtk.deid.kanonymity.KAnonymityReport

::: openbtk.deid.kanonymity.anonymise_to_k

::: openbtk.deid.kanonymity.AnonymisationResult

::: openbtk.deid.kanonymity.band

::: openbtk.deid.kanonymity.top_code

::: openbtk.deid.kanonymity.prefix

::: openbtk.deid.kanonymity.zip_ladder

::: openbtk.deid.kanonymity.identity

::: openbtk.deid.kanonymity.star

## Extending: recognizers

A recognizer finds candidate PHI spans; the engine merges them, filters by
confidence and transforms them. Implement `BaseRecognizer` and name it in
`DeidEngine(recognizers=[...])` to add your own detector.

::: openbtk.deid.recognizers.base.BaseRecognizer

::: openbtk.deid.recognizers.rule.RuleRecognizer
