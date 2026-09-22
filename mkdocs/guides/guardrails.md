# Guardrails

A guardrail checks one payload, usually a model's output, and returns a
`GuardrailResult`. It **never raises** for a failed check: `passed`, `severity`
(`info`, `warning`, `block`) and a message tell you what happened, and *you* decide
what a `block` means (drop the answer, regenerate, page someone).

```python
from openbtk.guardrails import GuardrailPipeline

pipeline = GuardrailPipeline(["guardrail.general.phi_leakage"])
result = pipeline.run("Call the patient on (555) 123-4567.")
print(result.passed, [r.severity.value for r in result.results])
```

## What is available

| Key | Checks | Needs |
|---|---|---|
| `guardrail.general.phi_leakage` | PHI in generated text | nothing |
| `guardrail.general.terminology` | codes exist in their declared system | a terminology service |
| `guardrail.general.groundedness` | claims are supported by the source | nothing (word overlap, see below) |
| `guardrail.general.dose_plausibility` | drug doses, daily totals and routes against **your** limits | a reference table you supply |
| `guardrail.ehr.code_validity` / `.referential` / `.units` | a `PatientRecord` | the `ehr` extra |
| `guardrail.ehr.k_anonymity` | a cohort cannot be narrowed below `k` people | the `ehr` extra |

## Dose plausibility

OpenBTK **ships no dose limits, on purpose.** A table of "safe" doses that nobody can
trace to a source is a patient-safety hazard, so the guardrail is an engine: it finds
dose statements in text and compares them with limits you provide from a formulary,
label or guideline you trust. Every `DoseLimit` must say where it came from.

The limits below are for a made-up drug and are **not clinical information**.

```python
from openbtk.guardrails.dose import DoseLimit, DosePlausibilityGuardrail

limits = [
    DoseLimit(
        drug="examplamine",
        unit="mg",
        min_single=5,
        max_single=100,
        max_daily=300,
        routes=["oral", "intravenous"],
        source="made-up values for this documentation only",
    )
]
guardrail = DosePlausibilityGuardrail(limits=limits)

ok = guardrail.check("Give examplamine 50 mg PO twice daily.")
bad = guardrail.check("Give examplamine 500 mg PO twice daily.")
print(ok.passed, bad.passed, bad.severity.value)
print(bad.details["problems"][0]["problem"])
```

For real use, keep the table in a file: a JSON list of the same fields. Pass its path
as `DosePlausibilityGuardrail(reference="limits.json")`; it is read on first use, not
when the guardrail is constructed.

```json
[
  {
    "drug": "examplamine",
    "aliases": ["exmpl"],
    "unit": "mg",
    "max_single": 100,
    "max_daily": 300,
    "routes": ["oral"],
    "source": "made-up values for this documentation only"
  }
]
```

What it does:

- reads a dose (`500 mg`, `0.5 g`, `5-10 mg`), an optional frequency (`BID`,
  `every 8 hours`) and an optional route (`PO`, `IV`) from the same clause as the
  drug name, and converts mass units;
- **blocks** a single dose or a computed daily total over a limit, and **warns** about
  a dose under the minimum, a route the reference does not list, or a unit it cannot
  compare (millilitres against milligrams, say);
- checks only the drugs in your table, and says so when none was mentioned.

What it does not do: it does not know the patient's weight, kidney function or other
medicines, and it cannot read a dose written in words. **A pass means "no limit you
supplied was broken by a statement it could read", never "this dose is safe".** With no
reference configured it reports that nothing was checked, as a warning, rather than
passing silently. A run manifest records how many limits were used and a hash of the
table, never its content.

## Combining guardrails

`GuardrailPipeline` runs several over one payload, in order, and can stop at the first
`block` (the default; `short_circuit=False` runs them all):

```python
both = GuardrailPipeline(
    ["guardrail.general.phi_leakage", DosePlausibilityGuardrail(limits=limits)],
    short_circuit=False,
)
combined = both.run("Give examplamine 500 mg PO. Call (555) 123-4567.")
print(combined.passed, len(combined.results))
```

To attach a guardrail to a step of a streaming pipeline instead, use
`Pipeline.guard` ([pipelines reference](../api/pipelines.md)).

## Groundedness is a heuristic

`guardrail.general.groundedness` checks that a generated sentence's content words
appear in the source. That catches an invented drug or number and misses a claim that
reuses the source's words to say something else. It is a filter, not a fact-checker. The
[evaluation guide](evaluation.md#groundedness) shows how to measure it against answers a
person has judged, which is how you should decide whether it is good enough for you.
