# Terminology

Clinical text says "CBC", "HTN" or "hypertenson"; the codes downstream want `58410-2`,
`I10`. OpenBTK's terminology layer does three jobs, and is careful about what it may
ship.

| Job | Class | Needs |
|---|---|---|
| Validate and resolve a **code** | `terminology.general.bundled_minimal`, `.local`, `.umls` | nothing / your CSV / your UMLS key |
| Match a free-text **term** to a concept | `ConceptNormalizer` | a vocabulary you supply |
| Cache lookups for offline use | `CachedTerminologyService` | a directory |

**OpenBTK bundles no licensed vocabulary.** The one that ships is a small ICD-10-CM
subset, which is public domain. LOINC, SNOMED CT and the rest are licensed, so they come
from you: as a CSV extract, or through your own UMLS account.

## Matching a term to a concept

`ConceptNormalizer` maps a term to the concepts of a vocabulary you give it, tolerating
case, punctuation, typos and the aliases you list. Whether "CBC" reaches `58410-2` is a
property of *your* vocabulary, which is where the alias lives:

```python
import contextlib
import io

with contextlib.redirect_stdout(io.StringIO()):
    from openbtk.core.schemas import CodeSystem, Concept
    from openbtk.terminology import ConceptNormalizer

vocabulary = [
    Concept(
        code="58410-2",
        system=CodeSystem.LOINC,
        display="Blood count panel (a display from your own licensed extract)",
    ),
    Concept(
        code="I10",
        system=CodeSystem.ICD10CM,
        display="Essential (primary) hypertension",
    ),
]
normalizer = ConceptNormalizer(
    vocabulary, aliases={"58410-2": ["CBC", "complete blood count"], "I10": ["HTN"]}
)

cbc = normalizer.resolve("CBC")
assert cbc is not None and cbc.concept.code == "58410-2" and cbc.method == "alias"

typo = normalizer.resolve("essential primary hypertenson")
assert typo is not None and typo.concept.code == "I10" and typo.method == "fuzzy"
```

A CSV with `code,system,display` and an optional `aliases` column (separated by `|`)
loads with `ConceptNormalizer.from_csv(path)`, so a licensed extract plus your own alias
list is a few lines of preparation, not code.

### It is conservative on purpose

A wrong clinical match is worse than none, so:

```python
normalizer = ConceptNormalizer(
    [
        Concept(
            code="E11.9",
            system=CodeSystem.ICD10CM,
            display="Type 2 diabetes mellitus without complications",
        ),
    ]
)

# Plain string similarity rates "type 1" and "type 2" as ~93% alike. Not here:
assert normalizer.resolve("Type 1 diabetes mellitus without complications") is None
# A shorter name is not the specific one: the code below says "without complications".
assert normalizer.resolve("Type 2 diabetes mellitus") is None
```

- A candidate must **agree on the words that change the meaning**: numbers, laterality
  (left, right), acuity (acute, chronic) and negation (no, without, non). The list is a
  safeguard, not a clinical ontology.
- Nothing below `threshold` (default 0.85) is returned. `resolve` gives `None`, not a
  guess, when the two best answers are different codes with nearly the same score;
  `normalise` returns the candidates with their scores so a person can decide.
- It matches a **term**, not a sentence. "No history of diabetes" is not a term for
  diabetes; run entity extraction and negation detection first.
- Scores are string similarity, not probabilities. No accuracy figure is claimed for any
  threshold: pick one by looking at your own data.

## Validating codes

```python
from openbtk.core.schemas import CodeSystem
from openbtk.terminology.bundled import BundledMinimalBackend

backend = BundledMinimalBackend()
assert backend.validate("E11.9", CodeSystem.ICD10CM)
assert backend.resolve("E11.9", CodeSystem.ICD10CM).display.startswith("Type 2")
assert backend.is_authoritative(CodeSystem.ICD10CM) is False  # a subset, so "unknown"
```

`is_authoritative` matters to the [guardrails](guardrails.md): a code missing from a small
subset is *unverifiable*, not *invalid*, and the terminology guardrail warns instead of
blocking. A local CSV is authoritative for the systems it contains, and UMLS for the systems it
serves.
