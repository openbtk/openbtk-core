# Tutorials

Eight Jupyter notebooks, in order. Each one runs **offline**, downloads no model,
uses only synthetic data, and is **executed in CI on every change** — a tutorial
that stopped working would fail the build.

The notebooks track `main`. Several use features and fixes newer than the 0.5.0
release (the command line, QA and groundedness evaluation, `DATE_SHIFT` beside
other identifiers, the terminology guardrail's warning-not-block behaviour), so
install from source until the next release.

To run them yourself:

```bash
pip install "openbtk[text,ehr,langchain,langgraph,notebooks] @ git+https://github.com/openbtk/openbtk-core.git" faiss-cpu
jupyter lab notebooks/
```

| # | Notebook | You will |
|---|---|---|
| 1 | [Your first pipeline](https://github.com/openbtk/openbtk-core/blob/main/notebooks/01_first_pipeline.ipynb) | load notes, de-identify, section and chunk them; read the run manifest; validate, run and replay from a config |
| 2 | [De-identification](https://github.com/openbtk/openbtk-core/blob/main/notebooks/02_deidentification.ipynb) | see all five transform modes and the audit report, see what the default recognizer misses, and score it on labelled data |
| 3 | [EHR to text](https://github.com/openbtk/openbtk-core/blob/main/notebooks/03_ehr_to_text.ipynb) | load FHIR patients, select a cohort, render a timeline as text, and de-identify it like a note |
| 4 | [Chunking for retrieval](https://github.com/openbtk/openbtk-core/blob/main/notebooks/04_chunking_for_retrieval.ipynb) | compare fixed-size and section-aware chunking on one note |
| 5 | [Retrieval with provenance](https://github.com/openbtk/openbtk-core/blob/main/notebooks/05_rag_with_provenance.ipynb) | index chunks in FAISS, rerank by shared concepts, and answer with a `SourceRef` to every source |
| 6 | [Guardrails and terminology](https://github.com/openbtk/openbtk-core/blob/main/notebooks/06_guardrails_and_terminology.ipynb) | check output for PHI and unsupported claims, and see what a partial vocabulary can and cannot say |
| 7 | [Evaluation](https://github.com/openbtk/openbtk-core/blob/main/notebooks/07_evaluation.ipynb) | measure retrieval, clinical QA accuracy and groundedness, and check the checker |
| 8 | [LangChain and LangGraph](https://github.com/openbtk/openbtk-core/blob/main/notebooks/08_langchain_and_langgraph.ipynb) | use OpenBTK components as Runnables, a vector store and a graph node |

The notebooks stand in for real models with two tiny offline providers (a hashing
"encoder" and an extractive "LLM") that implement OpenBTK's real interfaces, so
swapping in a real model changes a line or two. They demonstrate the *plumbing*;
they make no claim about retrieval or answer quality. For the numbers this
project does publish, and what they do not show, see [Benchmarks](benchmarks.md).
