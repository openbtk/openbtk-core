# OpenBTK — Open Toolkit for Biomedical AI

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**The open-source layer between biomedical data and modern AI.**

OpenBTK turns EHR and clinical text into model-ready, **de-identified**, **auditable**
inputs — and wraps model outputs in **clinical guardrails**. It runs on your laptop or
behind your firewall, works with any LLM provider, and emits the audit trail your
regulator and your IRB will ask for.

---

> ## 🚧 Status: pre-implementation
>
> **There is no working code yet, and nothing is published to PyPI.**
>
> The project is being rebuilt from scratch. A previous attempt produced ~5,200 lines
> that were never executable — three conflicting package names coexisted in one
> repository, and no test was ever run against an installed dependency. That code is
> preserved on the `legacy/v1-snapshot` branch and has been cleared from `dev`.
>
> The current phase — specification, architecture and planning — is complete.
> This README will list features when features exist, and not before.

---

## Why

Building a clinical RAG pipeline in 2026 means six weeks of undifferentiated plumbing:
a note parser, a PHI de-identifier that actually catches MRNs, chunking that doesn't
split the Assessment from the Plan, entity linking, code validation, and an audit
trail you'll be asked for later. Every team rebuilds it. Most get de-identification
wrong, or pay a vendor.

The existing tools each solve one slice. MONAI owns imaging. medspaCy owns rule-based
clinical text. PyHealth owns predictive modelling over EHR. LangChain owns
orchestration but treats a FHIR bundle and a blog post as the same thing. Commercial
platforms solve most of it, behind a licence that excludes academia and early-stage
teams.

**Nothing joins them, and nothing open-source addresses the two things that actually
gate deployment: is the PHI really gone, and can you prove what happened.**

Those two are what OpenBTK is for.

## Design commitments

- **Domain-aware by default** — chunking that respects clinical sections, retrieval
  that knows two strings can be the same SNOMED concept.
- **Wrap, don't reinvent** — thin, consistent adapters over medspaCy, scispaCy,
  presidio, `fhir.resources`; net-new code only where nothing adequate exists.
- **Streaming by default** — memory is `O(batch)`, not `O(corpus)`. Target: stream
  10M notes in under 4 GB RSS.
- **Light core** — six dependencies, no ML framework. Reading a clinical note should
  not require installing PyTorch.
- **Safe and provable by construction** — de-identification, guardrails and a run
  manifest are structural, not optional middleware.

## Planned scope

**v1 — clinical text and EHR/FHIR, taken to production quality.** Loading,
de-identification, section-aware chunking, entity linking, terminology resolution,
embedding and LLM providers, retrieval with concept reranking, clinical guardrails,
evaluation harness, run provenance, config-driven pipelines, CLI.

**v2 — imaging, biosignals, genomics, video, audio.** These will **wrap** MONAI, wfdb,
MNE, pysam and librosa rather than compete with them, and are gated until v1 ships
with published benchmarks.

**Never** — a model zoo, a serving platform, a clinical decision-support system, or a
dataset distributor.

## Documentation

Design documents — market research, PRD, architecture, API contract, security and
compliance, test charter, roadmap and ADRs — are maintained outside this repository
and are not published here. Public documentation ships with the v0.1 release.

## Contributing

Not yet open for contributions — the foundation is still being laid.
Contributions open at **v0.1**.

## License

Apache 2.0.

## Disclaimer

OpenBTK is a software toolkit. It is **not a medical device**, **not a clinical
decision-support system**, and provides **no medical advice**. It does not make its
users HIPAA- or EU AI Act-compliant — it provides technical controls supporting a
compliance programme its users own.

Datasets requiring credentialed access (MIMIC, eICU, n2c2, TCGA) are **not bundled**.
Users must obtain their own authorized access.
