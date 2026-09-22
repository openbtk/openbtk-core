# De-identification

`DeidEngine` finds protected health information (PHI) in text, transforms it,
and returns a `DeidReport` you can audit. Two things to read first:

!!! warning "What the default configuration does not do"
    The default recognizer (`rule`) finds **structured identifiers**: SSN, MRN,
    phone and fax numbers, e-mail, URLs, IP addresses, dates, and account /
    licence / device / plan numbers. It does **not** detect **names** or
    **street addresses**. Add the `ner` recognizer for those (`pip install
    "openbtk[text]"` and `python -m spacy download en_core_web_sm`); it finds most
    synthetic names but also over-flags, trading precision for recall. The
    [benchmarks](../benchmarks.md) page has the measured numbers and their limits.

!!! note "Not a compliance guarantee"
    OpenBTK provides technical controls. It does not make you HIPAA-compliant,
    and the published figures are on a synthetic corpus, not real clinical text.

## Detect, transform, report

```python
import contextlib
import io

with contextlib.redirect_stdout(io.StringIO()):  # registry logs, not errors
    from openbtk.deid import DeidEngine, DeidMode

    engine = DeidEngine(mode=DeidMode.REDACT, recognizers=["rule"])
    result = engine.deidentify(
        "Seen 03/14/2024. Call (555) 010-2345 to confirm.",
        patient_id="hashed-patient-id",
    )

assert result.text == "Seen [REDACTED]. Call [REDACTED] to confirm."
report = result.report
# The report holds categories, counts and offsets -- never the values found.
assert {c.value: n for c, n in report.entity_counts.items()} == {
    "date": 1,
    "phone_number": 1,
}
assert "010-2345" not in report.model_dump_json()
```

`patient_id` should already be a hashed or pseudonymous identifier; the engine
has no way to enforce that.

## Transform modes

| Mode | Replaces a span with | Use when |
|---|---|---|
| `REDACT` | `[REDACTED]` | you never need the value again |
| `TAG` | its category, e.g. `[PHONE_NUMBER]` | a downstream model should know something was there |
| `HASH` | a keyed hash | you need to tell two occurrences apart |
| `SURROGATE` | realistic fake data, consistent per patient | you want text that still reads naturally |
| `DATE_SHIFT` | each date moved by one per-patient offset (other identifiers are redacted) | intervals between events must survive |

```python
import contextlib
import io

with contextlib.redirect_stdout(io.StringIO()):
    from openbtk.deid import DeidEngine, DeidMode

    tagged = DeidEngine(mode=DeidMode.TAG).deidentify(
        "Call (555) 010-2345.", patient_id="p1"
    )

assert "[PHONE_NUMBER]" in tagged.text
```

## Measure it on your own data

Detection quality depends on your notes. Score the engine against spans you have
labelled, using the same scorer the project's CI gate uses:

```python
import contextlib
import io

with contextlib.redirect_stdout(io.StringIO()):
    from openbtk.deid import DeidEngine, DeidMode
    from openbtk.deid.labelled import LabelledDocument, LabelledSpan
    from openbtk.deid.schemas import PHICategory
    from openbtk.eval.deid import evaluate_deid, format_markdown

    text = "Seen 03/14/2024. Dr Jane Roe agrees."
    doc = LabelledDocument(
        document_id="d1",
        text=text,
        spans=[
            LabelledSpan(category=PHICategory.DATE, start=5, end=15),
            LabelledSpan(category=PHICategory.NAME, start=20, end=28),
        ],
    )
    scores = evaluate_deid(DeidEngine(mode=DeidMode.REDACT), [doc])

# The rule recognizer finds the date and (by design) misses the name.
assert scores.per_category[PHICategory.DATE].true_positives == 1
assert scores.per_category[PHICategory.NAME].false_negatives == 1
assert "date" in format_markdown(scores)
```

Matching is *relaxed*: a labelled span counts as found if any detection overlaps
it by at least one character, and a *binary* (any-PHI) view is reported next to
the category-aware one. It is a more forgiving rule than boundary-exact
scoring, so do not compare its numbers with papers that use a stricter one.

If you hold the i2b2/n2c2 2014 corpus under its data use agreement,
`python -m openbtk.eval.deid_benchmark --dataset n2c2 --path <dir>` runs the
same scorer over it. That harness is tested on hand-built files in the corpus's
format and has not been run against the real corpus.

## Structured data: k-anonymity

Removing names does not make a table of patients anonymous: a birth year, a sex and a
ZIP prefix together can single someone out. A table is **k-anonymous** over the columns
you name when every combination of their values is shared by at least `k` records.

```python
from openbtk.deid.kanonymity import anonymise_to_k, k_anonymity_report

table = [
    {"patient": f"p{i}", "birth_year": 1960 + i % 30, "gender": "f" if i % 2 else "m"}
    for i in range(120)
]
columns = ["birth_year", "gender"]

before = k_anonymity_report(table, columns, k=5)
print(before.k_achieved, before.n_unique, before.satisfied)

result = anonymise_to_k(table, columns, k=5, keep=["patient"])
print(result.report.satisfied, result.levels, result.n_suppressed)
print(result.rows[0])
```

The report holds **counts only**, never a value, so it is safe to log. `anonymise_to_k`
generalises the column with the most distinct values one step at a time (a year becomes a
5-year band, then a 10-year band, then `*`) and, only if a few records are still too rare,
suppresses them (`max_suppression`, default 5%). It returns exactly which steps it took.
Only the columns you name, plus any in `keep`, are exported; an unlisted column cannot
leak through. Ages are top-coded at 90, as Safe Harbor requires, and
`zip_ladder()` shortens a ZIP code.

For an EHR cohort, `quasi_identifiers` builds the table from `PatientRecord`s and
`guardrail.ehr.k_anonymity` checks a cohort directly:

```python
from openbtk.data.ehr.cohort import quasi_identifiers
from openbtk.data.ehr.schemas import Demographics, PatientRecord
from openbtk.guardrails.ehr import CohortKAnonymityGuardrail

cohort = [
    PatientRecord(
        patient_id=f"pt-{i}",
        demographics=Demographics(gender="female" if i % 2 else "male"),
        source_system="fhir-r4",
    )
    for i in range(12)
]
rows = quasi_identifiers(cohort, ["gender"])
print(k_anonymity_report(rows, ["gender"], k=5).satisfied)

check = CohortKAnonymityGuardrail(k=5, quasi_identifiers=["gender"]).check(cohort)
print(check.passed, check.message)
```

What this does **not** give you, and you should not claim it does:

- It protects only against linking on the columns you named. A rare diagnosis or a
  free-text note that you did not list is not protected.
- It says nothing about what a group *shares*: if every patient in a group has the same
  diagnosis, the diagnosis is disclosed.
- It does not choose `k`. Five is a common floor; the right value is a policy decision.
- The ZIP ladder does not know which three-digit prefixes are too small to publish (Safe
  Harbor lists them); apply that rule yourself.

It is evidence for an expert to weigh, not a de-identification determination.

## From the command line

`openbtk deid notes/ --out clean/` de-identifies a directory of notes without a
config; see the [CLI guide](cli.md).
