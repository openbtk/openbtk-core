# OpenBTK — Open Toolkit for Biomedical AI

[![CI](https://github.com/openbtk/openbtk/actions/workflows/ci.yml/badge.svg)](https://github.com/openbtk/openbtk/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/openbtk.svg)](https://pypi.org/project/openbtk/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**OpenBTK** is a production-grade Python framework for building AI pipelines
on biomedical data — EHR/EMR, clinical text, medical imaging, biosignals,
genomics, surgical video, and physiological audio.

It does for biomedical AI what LangChain/LangGraph did for general-purpose
LLM apps — except every component is **biomedical-domain-aware by default**,
wrapping mature libraries (`pydicom`, `scispacy`, `wfdb`, `MONAI`, `pysam`,
`LangGraph`) instead of reinventing them.

> ⚠️ **Status: Alpha.** APIs may change between minor versions until 1.0.

## Why OpenBTK?

General frameworks treat biomedical data as "just another document." They
have no concept of DICOM series, FHIR resources, ECG leads, VCF variants, or
PHI de-identification. OpenBTK provides composable, production-grade
building blocks for every step of a biomedical AI pipeline — loading,
preprocessing, chunking/segmenting, embedding, retrieval, guardrails, and
fine-tuning — with a consistent, configurable, provider-agnostic interface
across **every modality**.

## Supported Modalities

| Modality | Status | Wraps |
|---|---|---|
| Clinical Text | 🚧 In progress | scispacy, medspacy, presidio, transformers |
| EHR / EMR | 📋 Planned | fhir.resources, hl7apy, OMOP CDM |
| Medical Imaging | 📋 Planned | pydicom, SimpleITK, MONAI, openslide |
| Biosignals (ECG/EEG/EMG) | 📋 Planned | wfdb, mne, neurokit2, pyedflib |
| Genomics | 📋 Planned | pysam, cyvcf2, scanpy, biopython |
| Surgical Video | 📋 Planned | opencv, decord, av |
| Physiological Audio | 📋 Planned | librosa, torchaudio |

## Install

```bash
pip install openbtk                  # core only
pip install openbtk[clinical_text]   # + clinical NLP stack
pip install openbtk[imaging]         # + medical imaging stack
pip install openbtk[all]             # everything
```

## Quick Start

```python
from src.core.registry import (
    LOADER_REGISTRY, PREPROCESSOR_REGISTRY, CHUNKER_REGISTRY, GUARDRAIL_REGISTRY,
)
import src.data.clinical_text  # registers clinical text components

# Load → de-identify → section-segment → chunk → guardrail check
loader = LOADER_REGISTRY.create("loader.clinical_text.plain_text")
records = loader.load_all("./notes")

deidentify = PREPROCESSOR_REGISTRY.create(
    "preprocessor.clinical_text.deidentify", mode="surrogate"
)
segment = PREPROCESSOR_REGISTRY.create("preprocessor.clinical_text.section_segment")
chunker = CHUNKER_REGISTRY.create("chunker.clinical_text.section_aware", max_tokens=256)
phi_guard = GUARDRAIL_REGISTRY.create("guardrail.clinical_text.phi")

for record in records:
    record = deidentify.process(record)
    record = segment.process(record)
    chunks = chunker.chunk(record)
    for chunk in chunks:
        result = phi_guard.check(chunk.text)
        print(chunk.section, chunk.token_count, "PHI clean:", result.passed)
```

Config-driven pipelines (no code):

```yaml
name: clinical-text-rag
steps:
  - type: loader.clinical_text.plain_text
    params: { note_type: "Discharge Summary" }
  - type: preprocessor.clinical_text.deidentify
    params: { mode: "surrogate" }
  - type: chunker.clinical_text.section_aware
    params: { max_tokens: 512 }
  - type: embedding.clinical_text.pubmedbert
    params: { device: "cpu" }
```

## Documentation

- [Vision](docs/VISION.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Data Modality Specs](docs/DATA_MODALITY_SPEC.md)
- [API Design](docs/API_DESIGN.md)
- [Coding Standards](docs/CODING_STANDARDS.md)
- [Glossary](docs/GLOSSARY.md)
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)

## Contributing

This project is under active initial development. See
[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the task
breakdown and `CLAUDE.md` for AI-agent contribution context.

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Disclaimer

OpenBTK is a software toolkit, **not** a medical device and **not** a
clinical decision-support system. It does not provide medical advice.
Datasets requiring credentialed access (MIMIC, eICU, TCGA, etc.) are not
bundled — users must obtain their own authorized access.
