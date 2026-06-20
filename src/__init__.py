"""
openbtk — Open Toolkit for Biomedical AI

A production-grade Python framework for building biomedical AI applications.
Every component is biomedical-domain-aware by design.

Quick start:
    from openbtk.data.clinical_text import ClinicalTextLoader, SectionAwareChunker
    from openbtk.embeddings import PubMedBERTEmbedding
    from openbtk.pipelines import Pipeline, PipelineConfig

Install extras for each modality:
    pip install openbtk[clinical_text]
    pip install openbtk[imaging]
    pip install openbtk[biosignals]
    pip install openbtk[genomics]
    pip install openbtk[all]
"""
from __future__ import annotations

__version__ = "0.1.0"
__author__ = "openbtk Contributors"
__license__ = "Apache-2.0"

from openbtk.core.registry import (
    CHUNKER_REGISTRY,
    DATASET_REGISTRY,
    EMBEDDING_REGISTRY,
    FEATURE_EXTRACTOR_REGISTRY,
    FINETUNER_REGISTRY,
    GUARDRAIL_REGISTRY,
    LLM_REGISTRY,
    LOADER_REGISTRY,
    PREPROCESSOR_REGISTRY,
    SEGMENTER_REGISTRY,
    VECTORSTORE_REGISTRY,
    get_registry,
)
from openbtk.core.errors import (
    openbtkError,
    ConfigError,
    LoaderError,
    ProviderError,
    ProcessingError,
    GuardrailViolation,
)

__all__ = [
    "__version__",
    # Registries
    "LOADER_REGISTRY", "PREPROCESSOR_REGISTRY", "CHUNKER_REGISTRY",
    "SEGMENTER_REGISTRY", "FEATURE_EXTRACTOR_REGISTRY", "EMBEDDING_REGISTRY",
    "LLM_REGISTRY", "VECTORSTORE_REGISTRY", "GUARDRAIL_REGISTRY",
    "DATASET_REGISTRY", "FINETUNER_REGISTRY", "get_registry",
    # Errors
    "openbtkError", "ConfigError", "LoaderError", "ProviderError",
    "ProcessingError", "GuardrailViolation",
]
