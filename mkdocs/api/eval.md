# Evaluation

Every report holds counts and identifiers only, never document, question or
answer text. Published numbers and what they do and do not show are on the
[Benchmarks](../benchmarks.md) page; the evaluation guide is
[here](../guides/evaluation.md).

## De-identification

::: openbtk.eval.deid.evaluate_deid

::: openbtk.eval.deid.EvaluationResult

::: openbtk.eval.deid.CategoryMetrics

::: openbtk.eval.deid.report_dict

::: openbtk.eval.deid.format_markdown

::: openbtk.deid.labelled.LabelledDocument

::: openbtk.deid.labelled.LabelledSpan

::: openbtk.data.clinical_text.datasets.N2C2DeidDataset

## Retrieval

::: openbtk.eval.retrieval.evaluate_retrieval

::: openbtk.eval.retrieval.retriever_from

::: openbtk.eval.retrieval.RetrievalQuery

::: openbtk.eval.retrieval.RetrievalReport

::: openbtk.eval.retrieval.recall_at_k

::: openbtk.eval.retrieval.reciprocal_rank

::: openbtk.eval.retrieval.ndcg_at_k

## Clinical QA

::: openbtk.eval.qa.evaluate_qa

::: openbtk.eval.qa.LLMAnswerer

::: openbtk.eval.qa.QAReport

::: openbtk.eval.qa.MCQItem

::: openbtk.eval.qa.read_medqa_jsonl

::: openbtk.eval.qa.read_medmcqa_jsonl

::: openbtk.eval.qa.parse_choice

::: openbtk.eval.qa.format_prompt

::: openbtk.eval.qa.qa_manifest

## Groundedness

::: openbtk.eval.groundedness.score_groundedness

::: openbtk.eval.groundedness.evaluate_detector

::: openbtk.eval.groundedness.GroundedExample

::: openbtk.eval.groundedness.LabelledExample

::: openbtk.eval.groundedness.GroundednessReport

::: openbtk.eval.groundedness.DetectorReport

::: openbtk.eval.groundedness.groundedness_manifest

## Summaries: ROUGE and BERTScore

::: openbtk.eval.summarisation.evaluate_summaries

::: openbtk.eval.summarisation.rouge_scores

::: openbtk.eval.summarisation.BertScorer

::: openbtk.eval.summarisation.SummaryPair

::: openbtk.eval.summarisation.SummarisationReport

::: openbtk.eval.summarisation.summarisation_manifest

## Model cards

::: openbtk.eval.model_card.ModelCard

::: openbtk.eval.model_card.CardEvaluation

::: openbtk.eval.model_card.cards_from_run

## Manifests

::: openbtk.eval.manifest.EvalManifest

::: openbtk.eval.manifest.build_manifest

::: openbtk.eval.manifest.file_digest
