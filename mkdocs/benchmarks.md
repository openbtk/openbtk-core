# Benchmarks

OpenBTK's rule 14: **no accuracy number without a harness in the repository
that reproduces it.** This page lists every published de-identification
number, the exact command that regenerates it, and — just as importantly —
what has **not** been measured.

## What is and is not measured

| Corpus | Status |
|---|---|
| Seeded synthetic corpus (`tests/fixtures/labelled_phi_corpus.py`) | **Measured** — tables below |
| i2b2 / n2c2 2014 de-identification challenge | **Not run.** The corpus is released only under a Data Use Agreement and this project's CI and development environments do not have it. The harness and dataset adapter exist and are tested on hand-built files in the corpus's format; no i2b2/n2c2 number is published because none has been produced. |

!!! warning "Read the synthetic numbers for what they are"
    The synthetic corpus is generated from templates (Faker plus fixed
    formats) by the same project that wrote the recognizers, so a high score
    on it is a **regression gate and a sanity check, not evidence of
    real-world accuracy**. Real clinical text is messier than any template.
    Treat these figures as an upper bound, and expect lower numbers on real
    notes. Also note the two categories the default configuration cannot
    detect at all (`name`, `geographic_subdivision`): the default
    `DeidEngine` runs only the rule recognizer, so *names are not removed by
    default* — enable the NER recognizer (`recognizers=["rule", "ner"]`).

## Matching rule

Scoring is **relaxed entity-level overlap**, not the stricter boundary-exact
or token-level scoring some challenges use. A ground-truth span is found if
any detection overlaps it by at least one character. Two views are reported:

* **category-aware** — the detection must carry the same Safe Harbor category
  (per-category rows and the *overall* row, micro-averaged);
* **binary (any PHI)** — category ignored; was the span detected as anything?
  This is the leak-relevant view.

The implementation is `openbtk.eval.deid`, shared by the CI accuracy gate and
the n2c2 harness, so a published number and a regression gate cannot disagree.

## Default configuration: rule recognizer

Regenerate: `python tests/accuracy/report.py`

<!-- benchmark:rule:start -->
### Synthetic corpus, recognizers: rule

25 documents; matching: relaxed span overlap (any overlapping detection counts).

| Category | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| account_number | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| certificate_license_number | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| date | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| device_identifier | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| email | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| fax_number | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| geographic_subdivision | 0 | 0 | 25 | 1.000 | 0.000 | 0.000 |
| health_plan_beneficiary_number | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| ip_address | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| medical_record_number | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| name | 0 | 0 | 25 | 1.000 | 0.000 | 0.000 |
| other_unique_identifier | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| phone_number | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| ssn | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| url | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| vehicle_identifier | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| **overall (category-aware)** | 350 | 0 | 50 | 1.000 | 0.875 | 0.933 |
| **binary (any PHI)** | 350 | 0 | 50 | 1.000 | 0.875 | 0.933 |
<!-- benchmark:rule:end -->

## Full ensemble: rule + NER

Regenerate: `python tests/accuracy/report.py --recognizers rule,ner`
(requires `pip install "openbtk[text]"` and
`python -m spacy download en_core_web_sm`).

<!-- benchmark:ner:start -->
### Synthetic corpus, recognizers: rule + ner

25 documents; matching: relaxed span overlap (any overlapping detection counts).

| Category | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| account_number | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| certificate_license_number | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| date | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| device_identifier | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| email | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| fax_number | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| geographic_subdivision | 2 | 0 | 23 | 1.000 | 0.080 | 0.148 |
| health_plan_beneficiary_number | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| ip_address | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| medical_record_number | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| name | 24 | 40 | 1 | 0.375 | 0.960 | 0.539 |
| other_unique_identifier | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| phone_number | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| ssn | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| url | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| vehicle_identifier | 25 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| **overall (category-aware)** | 376 | 40 | 24 | 0.904 | 0.940 | 0.922 |
| **binary (any PHI)** | 389 | 27 | 11 | 0.935 | 0.973 | 0.953 |
<!-- benchmark:ner:end -->

The ensemble trades precision for recall: names are now found (recall 0.960)
but a general-purpose spaCy model also tags structured `Label: value` text as
people, so name precision falls to 0.375. Geographic subdivisions barely
improve (recall 0.080). A clinically-tuned NER model would very likely do
better; that has not been measured here.

## Running the n2c2 / i2b2 benchmark yourself

If you hold the i2b2/n2c2 2014 de-identification data under its Data Use
Agreement:

```bash
python -m openbtk.eval.deid_benchmark --dataset n2c2 --path /path/to/n2c2/xml     --recognizers rule,ner --json report.json --markdown report.md
```

The reports contain counts and category names only — never document text —
so they are safe to share. The XML reader was written from the published
annotation scheme and **has not been validated against the real corpus**; it
refuses (rather than mis-scores) files whose tag offsets do not match their
text. If it does not read your copy, that is a bug worth reporting.

## Retrieval

`openbtk.eval.retrieval` provides recall@k, MRR and nDCG@k
(`evaluate_retrieval`, `retriever_from`). No retrieval benchmark number is
published: there is no agreed corpus in this repository yet, and rule 14
forbids a number without one.
