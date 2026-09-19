# OpenBTK

**The open-source layer between biomedical data and modern AI.**

OpenBTK turns clinical text into model-ready, **de-identified**, **auditable**
inputs — de-identification you can audit and benchmark, and run provenance
every pipeline execution emits automatically. It runs on your laptop or
behind your firewall, works with any LLM provider, and streams corpora far
larger than memory.

!!! note "Status: pre-1.0, real, and honestly scoped"
    Milestones **M1–M8** are built, tested, and merged: core framework,
    de-identification, clinical text, EHR loading, providers and retrieval,
    guardrails and terminology, benchmarks, and LangChain interop. **PyPI
    carries `0.1.1`** (M1–M3); the rest is on `main` and ships with the next
    release. See [Quick start](quickstart.md) for install instructions.

!!! warning "One published number is missing, on purpose"
    The i2b2/n2c2 de-identification benchmark harness exists and is tested,
    but that corpus is Data-Use-Agreement-restricted and has **not** been run.
    The published figures are on a synthetic corpus. See
    [Benchmarks](benchmarks.md).

## Why

Building a clinical RAG pipeline means weeks of undifferentiated plumbing: a
note parser, a PHI de-identifier that actually catches MRNs, chunking that
doesn't split the Assessment from the Plan, and an audit trail you'll be
asked for later. Every team rebuilds it. Most get de-identification wrong,
or pay a vendor.

**Nothing open-source addresses the two things that actually gate
deployment: is the PHI really gone, and can you prove what happened.** Those
two are what OpenBTK is for.

## What's built

- **A registry-driven component framework** — loaders, preprocessors,
  chunkers, guardrails and more, addressed by a permanent string key
  (`<category>.<scope>.<name>`), config-driven rather than import-driven.
- **De-identification you can audit and benchmark** — a rule-based
  recognizer plus an optional spaCy-based NER recognizer, combined by an
  ensemble with checked-in, measured F1 numbers (0.933 default, 0.922 with
  the full ensemble).
- **A clinical text modality** — loaders, section segmentation, and
  section-aware chunking that never splits mid-section.
- **A streaming pipeline executor with real provenance** — every run emits
  a `RunManifest`: per-step record counts, component identity, guardrail
  outcomes, and a content digest for each input source.
- **Streaming, not in-memory** — 10,000,000 synthetic notes streamed
  through the full four-stage pipeline at 0.056 GB peak RSS, measured.
- **EHR loading** — FHIR R4 and OMOP into one `PatientRecord` schema, with
  timelines, cohorts, and a serializer bridging EHR to clinical text.
- **Providers and retrieval** — LLM and embedding providers behind one
  interface with an off-site policy guard; FAISS, Chroma and Qdrant stores;
  concept reranking; a RAG pipeline with source provenance.
- **Guardrails and terminology** — PHI-leakage, terminology, groundedness
  and EHR checks; a terminology service with a bundled ICD-10-CM subset.
- **Evaluation and interop** — [reproducible benchmarks](benchmarks.md),
  retrieval metrics, and an optional [LangChain / LangGraph
  adapter](langchain.md).

See the [project README](https://github.com/openbtk/openbtk-core) and
[CHANGELOG](https://github.com/openbtk/openbtk-core/blob/main/CHANGELOG.md)
for the full, dated record of what has shipped.

## Where to go next

- [Quick start](quickstart.md) — install and run a real pipeline.
- [API reference](api/core.md) — generated from the source docstrings, so it
  never claims more than what actually ships.
