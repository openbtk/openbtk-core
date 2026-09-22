# Evaluation

OpenBTK ships harnesses, not scores. Every number this project publishes comes
from a harness in the repository (see [Benchmarks](../benchmarks.md)); the
harnesses below let you produce the same kind of number for *your* model and
*your* data. Reports hold counts and identifiers, never text, so they are safe
to file.

| You want to measure | Use |
|---|---|
| how well de-identification finds PHI | `openbtk.eval.deid.evaluate_deid` ([guide](deidentification.md)) |
| whether retrieval surfaces the right chunks | `openbtk.eval.retrieval.evaluate_retrieval` |
| a model's accuracy on multiple-choice clinical questions | `openbtk.eval.qa.evaluate_qa` |
| how faithful generated text is to its sources | `openbtk.eval.groundedness.score_groundedness` |

## Retrieval: recall@k, MRR, nDCG@k

```python
from openbtk.eval.retrieval import RetrievalQuery, evaluate_retrieval

queries = [
    RetrievalQuery(
        query_id="q1", query="statin therapy", relevance={"c1": 1.0, "c2": 1.0}
    ),
    RetrievalQuery(query_id="q2", query="insulin dosing", relevance={"c3": 1.0}),
]
# A retriever is any function from a query to a ranked list of ids.
ranked = {"statin therapy": ["c1", "c9", "c2"], "insulin dosing": ["c7", "c3"]}
report = evaluate_retrieval(lambda q: ranked[q], queries, ks=(1, 3))

assert report.n_queries == 2
assert report.recall_at_k[1] == 0.25  # (1/2 relevant found + 0/1) / 2 queries
assert report.recall_at_k[3] == 1.0
assert report.mrr == 0.75  # ranks 1 and 2 -> (1 + 1/2) / 2
```

`retriever_from(embedding, vector_store)` builds such a function from a real
embedding provider and vector store.

## Clinical QA (MedQA / MedMCQA format)

`evaluate_qa` scores any function from a prompt to a reply, so a real model is
one adapter away. This example uses a stand-in "model" so it runs anywhere:

```python
import json
import pathlib
import tempfile

from openbtk.eval.qa import evaluate_qa, read_medmcqa_jsonl

rows = [  # invented questions, in MedMCQA's file format
    {
        "id": "q1",
        "question": "Which vitamin deficiency causes scurvy?",
        "opa": "Vitamin A",
        "opb": "Vitamin C",
        "opc": "Vitamin D",
        "opd": "Vitamin K",
        "cop": 1,
        "subject_name": "Biochemistry",
    },
    {
        "id": "q2",
        "question": "Which organ produces insulin?",
        "opa": "Liver",
        "opb": "Kidney",
        "opc": "Pancreas",
        "opd": "Spleen",
        "cop": 2,
        "subject_name": "Physiology",
    },
]
folder = tempfile.TemporaryDirectory()
path = pathlib.Path(folder.name, "validation.jsonl")
path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

replies = iter(["B", "The answer is A."])  # right, then wrong
report = evaluate_qa(lambda prompt: next(replies), read_medmcqa_jsonl(path))

assert (report.n, report.correct, report.unanswered) == (2, 1, 0)
assert report.accuracy == 0.5
low, high = report.accuracy_ci95  # 2 questions: a very wide interval
assert low < 0.5 < high
folder.cleanup()
```

To score a real model, wrap a provider: `evaluate_qa(LLMAnswerer(provider), items)`.
Points to keep straight:

* **No dataset and no score is bundled.** `read_medqa_jsonl` and
  `read_medmcqa_jsonl` read files you have downloaded from the public sources.
  MedMCQA's public *test* split hides its answers, so that file is refused; score
  the validation split.
* **A reply that names no option counts as wrong** and is also reported as
  `unanswered`, so a model that rambles is visible rather than merely low.
* **Read the interval, not just the accuracy.** `accuracy_ci95` is a Wilson 95%
  interval; on a few hundred questions it is several points wide.
* `qa_manifest(report, component=provider.provenance(), source=path)` records
  which model, which file (by SHA-256) and when, as an `EvalManifest`.

## Groundedness

```python
from openbtk.eval.groundedness import (
    GroundedExample,
    LabelledExample,
    evaluate_detector,
    score_groundedness,
)

context = ["Assessment: type 2 diabetes mellitus, stable. Metformin continued."]
answers = [
    GroundedExample(
        example_id="a", answer="The patient has type 2 diabetes.", context=context
    ),
    GroundedExample(
        example_id="b", answer="The patient has a fractured femur.", context=context
    ),
]
report = score_groundedness(answers)
assert report.faithfulness == 0.5  # 1 of 2 claims supported

# How good is the checker itself? Score it against answers a human has judged.
judged = [
    LabelledExample(
        example_id="a", answer=answers[0].answer, context=context, grounded=True
    ),
    LabelledExample(
        example_id="b", answer=answers[1].answer, context=context, grounded=False
    ),
]
assert evaluate_detector(judged).f1 == 1.0
```

!!! warning "The default checker is a word-overlap heuristic"
    It measures lexical overlap with the context. It cannot see a negation, a
    swapped dose, or a paraphrase, so a high faithfulness score does not mean a
    clinically faithful answer. Build a `GroundednessGuardrail` with your own
    `is_supported` (an entailment model or an LLM judge) for a stronger check, and
    run `evaluate_detector` on your own labelled examples before trusting any
    faithfulness number.

## Model cards

A model card is the short document that travels with a model: what it is, what it is for,
how well it does, where it fails. OpenBTK already records the facts a card needs, so it
assembles them: the exact model and pinned revision a component used, and the scores an
evaluation produced.

```python
from datetime import UTC, datetime

from openbtk.eval.manifest import EvalManifest
from openbtk.eval.model_card import ModelCard
from openbtk.retrieval.cross_encoder import CrossEncoderReranker

card = ModelCard.for_component(
    CrossEncoderReranker(),
    intended_use="Reordering retrieved passages for clinical question answering.",
    limitations="Not evaluated on non-English text or on notes over 512 tokens.",
)

# A score reaches a card only from an evaluation run's manifest.
run = EvalManifest(
    eval_id="retrieval-2026-09",
    kind="retrieval",
    started_at=datetime(2026, 9, 1, tzinfo=UTC),
    ended_at=datetime(2026, 9, 1, tzinfo=UTC),
    report={"recall_at_5": 0.5},
)
card = card.with_evaluation(run, "recall_at_5", name="Recall@5")

text = card.to_markdown()
assert "ncbi/MedCPT-Cross-Encoder" in text
assert "| Recall@5 | 0.5 | `retrieval-2026-09` (retrieval) |" in text
assert "## Training data\n\nNot provided." in text  # nothing invented
```

What is written for you, and what is not:

- The **identity** (model, pinned revision, source, the OpenBTK component and its
  settings) comes from the component's provenance, and the **evaluation table** from
  `EvalManifest`s you attach, each with its id and input digests.
- Everything that needs judgement (intended use, out-of-scope use, training data,
  limitations, ethical considerations) is text *you* provide. A section you leave out
  reads "Not provided." A card never describes a model's training data, invents
  limitations or praises it.
- **There is no way to type a score into a card.** `with_evaluation` reads a number from a
  manifest's report and refuses one that is not there, so no figure appears without the
  run that produced it. A card with no evaluation says so.
- For a whole run, `cards_from_run(manifest)` writes one card per distinct model the run
  used. For a model you fine-tuned, build a `ModelCard` from its `ModelIdentity` and say
  what you trained it on.

## Summaries: ROUGE and BERTScore

For a discharge summary or similar, `evaluate_summaries` compares each candidate with a
reference on ROUGE (word, word-pair and longest-common-subsequence overlap) and, if you
give it a scorer, BERTScore (similarity in a language model's embedding space, which credits
a paraphrase that ROUGE misses).

```python
from openbtk.eval.summarisation import SummaryPair, evaluate_summaries

pairs = [
    SummaryPair(
        example_id="d1",
        reference="the patient is stable today",
        candidate="the patient is stable",
    ),
]
report = evaluate_summaries(pairs, use_stemmer=False)

# The candidate has 4 of the reference's 5 words and nothing extra:
assert report.rouge["rouge1"].precision == 1.0
assert report.rouge["rouge1"].recall == 0.8
assert report.as_dict()["n"] == 1
```

Read these numbers for what they are:

- They measure **similarity to a reference, not correctness.** A fluent summary that
  leaves out the one abnormal result can score well, and a faithful paraphrase can score
  badly. Compare systems on your own data with them, next to the groundedness check above;
  do not use them as a safety measure.
- ROUGE wraps Google's `rouge-score` (the `eval` extra). Its tokenizer keeps only `a-z`
  and `0-9`, so accented letters and symbols such as `%` or `>` are dropped before
  scoring, a real limit for clinical text.
- **BERTScore needs choices OpenBTK will not make for you.** You give the model, its
  pinned commit *and* the layer, and it downloads nothing until first use:

```python
from openbtk.eval.summarisation import BertScorer

scorer = BertScorer(
    model="distilbert-base-uncased",
    revision="12040accade4e8a0f71eabdb258fecc2e7e948be",  # pragma: allowlist secret
    layer=5,
)
assert scorer.provenance().model_identity.revision.startswith("12040acc")
# evaluate_summaries(pairs, bertscorer=scorer)  # downloads the model, needs torch
```

  OpenBTK implements BERTScore itself rather than wrapping the `bert-score` package,
  because that package fetches its model by name with no way to pin a revision. The
  implementation reproduces the reference package's raw scores (five sentence pairs agree
  to within 1e-7; the test that checks this carries the recorded values), but supports
  neither idf weighting nor baseline rescaling, so its numbers are comparable only with
  other *raw* scores from the same model and layer.
- The report holds scores and counts, never text. `summarisation_manifest(report,
  bertscorer=scorer)` records when it ran, the scores, and the BERTScore model, so a score
  can go on a [model card](#model-cards) through its manifest.
