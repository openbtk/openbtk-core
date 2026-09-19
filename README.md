# OpenBTK — Open Toolkit for Biomedical AI

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**The open-source layer between biomedical data and modern AI.**

OpenBTK turns clinical text into model-ready, **de-identified**, **auditable**
inputs — de-identification you can audit and benchmark, and run provenance
every pipeline execution emits automatically. It runs on your laptop or behind
your firewall, works with any LLM provider, and streams corpora far larger
than memory.

---

> ## Status: v0.5 — pre-1.0, real, and honestly scoped
>
> Milestones **M1–M9** are built, tested and released: the core framework,
> de-identification, clinical text, EHR loading, providers and retrieval,
> guardrails and terminology, benchmarks, and LangChain interop.
> `pip install openbtk` gives you `0.5.0`. `main` also has the `openbtk` command
> line, clinical-QA and groundedness evaluation, guides and eight tutorial
> notebooks (M10), which ship with the next release.
>
> **The i2b2/n2c2 de-identification benchmark has not been run.** The harness
> exists and is tested, but that corpus is released only under a Data Use
> Agreement. Every number below is on a **synthetic** corpus — a regression
> gate, not evidence of real-world accuracy. The default recognizer does not
> detect names. See [Benchmarks](mkdocs/benchmarks.md).
>
> A previous attempt (`legacy/v1-snapshot`) produced ~5,200 lines that were
> never executable. Every claim below is backed by a test that runs in CI —
> see [CHANGELOG.md](CHANGELOG.md) for the full, dated record.

---

## Why

Building a clinical RAG pipeline means weeks of undifferentiated plumbing: a
note parser, a PHI de-identifier that actually catches MRNs, chunking that
doesn't split the Assessment from the Plan, and an audit trail you'll be
asked for later. Every team rebuilds it. Most get de-identification wrong, or
pay a vendor.

**Nothing open-source addresses the two things that actually gate
deployment: is the PHI really gone, and can you prove what happened.** Those
two are what OpenBTK is for.

## What's built

- **A registry-driven component framework** — loaders, preprocessors,
  chunkers, guardrails and more, addressed by a permanent string key
  (`<category>.<scope>.<name>`), config-driven rather than import-driven.
- **De-identification you can audit and benchmark**: a rule-based recognizer
  (SSN, MRN, email, phone, dates, and eight more HIPAA Safe Harbor
  categories) plus an optional spaCy-based NER recognizer for names and
  addresses, combined by an ensemble that measurably protects the
  rule-covered categories from regressing when NER is added. Every claim is
  a checked-in number, not an estimate:

  | Configuration | Overall F1 | Notes |
  |---|---|---|
  | Default (`recognizers=["rule"]`) | **0.933** | Zero extras required |
  | Full ensemble (`+ NER`) | **0.922** | Name recall 0.0 → 0.96; trades precision (~0.38) — disclosed, not hidden |

  Four transform modes (`REDACT`, `TAG`, `HASH`, `SURROGATE`) plus
  interval-preserving date shifting, verified with a Hypothesis property
  test. See [`tests/accuracy/`](tests/accuracy/).
- **A clinical text modality**: loaders for plain text, JSON Lines and MIMIC-
  shaped CSV; rule-based and medspaCy-backed section segmentation;
  section-aware chunking that never splits mid-section (the one place
  OpenBTK writes genuinely new logic — see
  [`src/openbtk/data/clinical_text/chunking.py`](src/openbtk/data/clinical_text/chunking.py)).
- **A streaming pipeline executor with real provenance**: compose steps with
  `Pipeline`/`Step` or a YAML config, and every run emits a `RunManifest` —
  per-step record counts, component identity, guardrail outcomes, and a
  content digest for each input source. No manifest-off switch. Verified
  end to end against a labelled synthetic PHI corpus: no identifier from the
  corpus appears anywhere in a serialised manifest.
- **Streaming, not in-memory**: **10,000,000 synthetic notes streamed through
  the full four-stage pipeline (load → de-identify → segment → chunk) at
  0.074 GB peak RSS** — measured, not projected. See
  [`tests/benchmark/test_memory.py`](tests/benchmark/test_memory.py).
- **EHR loading**: FHIR R4 (`Bundle` and per-resource, streamed) and OMOP
  (core tables in `pyarrow` batches) into one `PatientRecord` schema, with
  encounter-anchored timelines, composable cohort predicates, and a
  serializer that turns a patient timeline into text the clinical-text side
  can chunk and embed. Exercised by a Synthea-shaped round-trip test (needs
  `pip install "openbtk[ehr]"`).
- **LLM and embedding providers behind one interface**: OpenAI, Anthropic,
  HuggingFace-local and any OpenAI-compatible endpoint, plus configuration
  presets for MedGemma, Meditron and OpenBioLLM; embeddings from
  HuggingFace models (presets for PubMedBERT, BioBERT, ClinicalBERT, SapBERT
  and MedCPT) and OpenAI. Every provider declares whether it sends data
  off-site, and a local-only policy refuses to construct one that does.
  *In CI the cloud SDKs and models are mocked; calls against live services
  run only when you supply keys or models.*
- **Retrieval**: FAISS, Chroma and Qdrant stores, a UMLS-concept-overlap
  reranker, and a RAG pipeline whose answers carry `SourceRef` provenance
  back to the originating record.
- **Guardrails and terminology**: PHI-leakage, terminology-validity,
  groundedness (a disclosed word-overlap heuristic, not entailment) and EHR
  integrity/units checks that always return a result and never raise on a
  failed check; a terminology service with a bundled ICD-10-CM subset, local
  vocabularies and UMLS (bring your own licence; the UMLS mapping call has
  not been verified against the live service).
- **Evaluation and interop**: de-identification scoring shared by the CI
  gate and the credentialed benchmark; recall@k, MRR and nDCG@k; and an
  optional LangChain/LangGraph adapter that is tested against the real
  libraries while the rest of the suite is verified to pass without them.

## Quick start

```bash
pip install "openbtk[text] @ git+https://github.com/openbtk/openbtk-core.git"
```

```python
import contextlib
import io
import tempfile
from pathlib import Path

from openbtk.data import clinical_text  # noqa: F401 -- registers loaders/preprocessors/chunkers
from openbtk.pipelines import Pipeline, Step

with tempfile.TemporaryDirectory() as notes_dir:
    (Path(notes_dir) / "note1.txt").write_text(
        "Chief Complaint:\n"
        "Patient SSN: 123-45-6789 reports chest pain.\n"
        "Plan:\n"
        "Admit for observation.\n"
    )

    pipeline = (
        Pipeline("clinical-rag")
        .add(Step("load", "loader.clinical_text.plain_text", path=notes_dir))
        .add(Step("deid", "preprocessor.general.deidentify", mode="surrogate"))
        .add(Step("segment", "preprocessor.clinical_text.section_segment"))
        .add(Step("chunk", "chunker.clinical_text.section_aware", max_tokens=200))
    )
    with contextlib.redirect_stdout(io.StringIO()):  # structured JSON logs, not errors
        manifest = pipeline.run()

assert manifest.status == "success"
for step in manifest.steps:
    print(f"{step.step_id}: {step.records_out} record(s)")
# load: 1 record(s)
# deid: 1 record(s)
# segment: 1 record(s)
# chunk: 2 record(s)
```

Every submodule you use (here, `openbtk.data.clinical_text`) must be
imported explicitly — `import openbtk` alone stays deliberately light (no
modality module, no heavy dependency, < 500 ms) so `pip install openbtk`
with zero extras still works. This is documented behaviour, in
[`openbtk/__init__.py`](src/openbtk/__init__.py)'s own module docstring, not
an oversight.

This exact example — extracted from this README and executed — is what CI
runs on every change; see [`tests/unit/test_readme.py`](tests/unit/test_readme.py).

Guardrails and terminology need nothing beyond the core install either. A
guardrail returns a result rather than raising, so a pipeline decides what a
failure means:

```python
import contextlib
import io

with contextlib.redirect_stdout(io.StringIO()):  # structured JSON logs, not errors
    from openbtk.core.schemas import CodeSystem
    from openbtk.guardrails.groundedness import (
        GroundednessCheckInput,
        GroundednessGuardrail,
    )
    from openbtk.guardrails.phi_leakage import PHILeakageGuardrail
    from openbtk.terminology.bundled import BundledMinimalBackend

    leak = PHILeakageGuardrail().check("Contact the patient at (555) 010-2345.")
    answer = GroundednessCheckInput(
        answer="The patient has a fractured femur.",
        context=["Assessment: type 2 diabetes mellitus, stable."],
    )
    grounded = GroundednessGuardrail().check(answer)
    concept = BundledMinimalBackend().resolve("E11.9", CodeSystem.ICD10CM)

assert not leak.passed  # a phone number is PHI; the message names the category only
assert not grounded.passed  # the claim is not supported by the context
assert concept is not None and concept.display.startswith("Type 2 diabetes")
```

## Installing

| What you get | Command |
|---|---|
| Core + clinical text (loaders, de-id, chunking, pipelines) | `pip install "openbtk[text]"` |
| Everything | `pip install "openbtk[text,ehr,retrieval,llms,langchain]"` |
| Core only (registry, config, provenance — no modality) | `pip install openbtk` |
| Latest `main` from source | `pip install "openbtk[text] @ git+https://github.com/openbtk/openbtk-core.git"` |

Optional extras: `text` (NER, medspaCy, tokenizers), `ehr` (FHIR, OMOP),
`retrieval` (FAISS, Chroma, Qdrant), `llms` (OpenAI, Anthropic, local
models), `langchain` (LangChain adapter), `langgraph` (to run LangGraph
graphs). Each component that needs one says so, by name, when you construct
it.

`openbtk[text]`'s optional NER recognizer additionally needs a downloaded
spaCy model: `python -m spacy download en_core_web_sm`.

## Design commitments

- **Domain-aware by default** — chunking that respects clinical sections.
- **Wrap, don't reinvent** — thin, consistent adapters over medspaCy and
  spaCy; net-new code only where nothing adequate exists (chunking is the
  main example so far).
- **Streaming by default** — memory is `O(batch)`, not `O(corpus)`. Measured:
  10M notes in 0.074 GB RSS, against a 4 GB target.
- **Light core** — six runtime dependencies, no ML framework in core.
  Reading a clinical note does not require installing PyTorch.
- **Safe and provable by construction** — de-identification and run
  provenance are structural, not optional middleware.

## Roadmap

**Next (M11):** freeze the public API and prepare v1.0. The one open v0.5 gate — the
published i2b2/n2c2 de-identification result (data access needed; see above)
— is audited in [`tests/release/`](tests/release/).

**Later:** clinical QA and groundedness scoring, and clinical entity
linking (ConText) for clinical text.

**v2 and beyond — imaging, biosignals, genomics, video, audio.** These will
**wrap** MONAI, wfdb, MNE, pysam and librosa rather than compete with them.
They stay gated until the current scope ships with published benchmarks —
including the n2c2 result above.

**Never** — a model zoo, a serving platform, a clinical decision-support
system, or a dataset distributor.

## Documentation

The site is live at <https://openbtk.org/openbtk-core/> (the `dev` build of
`main`).

- [Tutorial notebooks](notebooks/) — eight, offline, executed in CI.
- [Guides and API reference](https://openbtk.org/openbtk-core/dev/) — clinical text,
  de-identification, EHR, evaluation, the command line; every code example on the
  guides is executed by a test.
- [Benchmarks](mkdocs/benchmarks.md) — every published number, its
  reproduction command, and what has *not* been measured.
- [LangChain and LangGraph](mkdocs/langchain.md) — the optional adapter; every
  code block on the page is executed by a test.
- [Quick start](mkdocs/quickstart.md) and the generated API reference in
  `mkdocs/`.

The design documents — market research, PRD, architecture, API contract,
security and compliance, test charter, roadmap and ADRs — are maintained
outside this repository today.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up a development
environment, run the test suite, and open a pull request. Please also read
the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Disclaimer

OpenBTK is a software toolkit. It is **not a medical device**, **not a
clinical decision-support system**, and provides **no medical advice**. It
does not make its users HIPAA- or EU AI Act-compliant — it provides
technical controls supporting a compliance programme its users own.

Datasets requiring credentialed access (MIMIC, eICU, n2c2, TCGA) are **not
bundled**. Users must obtain their own authorized access.
