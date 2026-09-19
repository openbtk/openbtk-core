# Tutorial notebooks

| # | Notebook |
|---|---|
| 1 | [Your first pipeline](01_first_pipeline.ipynb) |
| 2 | [De-identification](02_deidentification.ipynb) |
| 3 | [EHR to text](03_ehr_to_text.ipynb) |
| 4 | [Chunking for retrieval](04_chunking_for_retrieval.ipynb) |
| 5 | [Retrieval with provenance](05_rag_with_provenance.ipynb) |
| 6 | [Guardrails and terminology](06_guardrails_and_terminology.ipynb) |
| 7 | [Evaluation](07_evaluation.ipynb) |
| 8 | [LangChain and LangGraph](08_langchain_and_langgraph.ipynb) |

Every notebook runs offline, downloads no model, and uses only fictitious data.
They need OpenBTK 0.6.0 or later.

```bash
pip install "openbtk[text,ehr,langchain,langgraph,notebooks]" faiss-cpu
jupyter lab notebooks/
```

## How they are kept honest

`tests/notebooks/` runs each notebook top to bottom in a fresh kernel and fails
on any error, and CI's `test-notebooks` job does so with every extra installed and
fails if any notebook is skipped. The committed outputs are from a real run; they
contain no machine-specific paths.

## Editing one

Edit it in Jupyter, run it top to bottom (Kernel → Restart & Run All) so its stored
outputs are current, then run `ruff format notebooks` and `ruff check notebooks`
(CI lints notebooks too). A notebook that writes files must first
`os.chdir(tempfile.mkdtemp())` so running it does not litter this folder.
