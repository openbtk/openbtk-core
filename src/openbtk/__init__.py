"""OpenBTK -- the open-source layer between biomedical data and modern AI.

Turns EHR and clinical text into model-ready, de-identified, auditable inputs,
and wraps model outputs in clinical guardrails.

This module is deliberately minimal. It must import in under 500ms with zero
optional dependencies installed, so it pulls in no modality module and nothing
heavy. Reach everything else through submodules::

    from openbtk.deid import DeidEngine
    from openbtk.data.clinical_text import SectionAwareChunker

Public API surface is defined in docs/04_API_DESIGN.md. Registry keys are part
of that surface and are permanent once released.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from openbtk.core.errors import OpenBTKError
from openbtk.core.registry import (
    CHUNKER_REGISTRY,
    DATASET_REGISTRY,
    EMBEDDING_REGISTRY,
    # FEATURE_EXTRACTOR_REGISTRY is not in docs/04_API_DESIGN.md section 2's
    # illustrative export list, but BaseFeatureExtractor is a PRD-required
    # base class (FR-C-03) with no principled reason to be less accessible
    # than its siblings. Included for consistency; documented here since it
    # is a deliberate deviation from that snippet, not an omission.
    FEATURE_EXTRACTOR_REGISTRY,
    GUARDRAIL_REGISTRY,
    LLM_REGISTRY,
    LOADER_REGISTRY,
    PREPROCESSOR_REGISTRY,
    RERANKER_REGISTRY,
    SEGMENTER_REGISTRY,
    TERMINOLOGY_REGISTRY,
    VECTORSTORE_REGISTRY,
    get_registry,
)

# PipelineConfig/StepConfig (core/config.py, task 1.8) and RunManifest
# (core/provenance.py's second increment) are not yet implemented -- see
# core/provenance.py's module docstring for why RunManifest is deferred.
# Add their imports and __all__ entries here once those modules exist; do
# not add placeholder names to __all__ for things that do not exist yet.

__all__ = [
    "CHUNKER_REGISTRY",
    "DATASET_REGISTRY",
    "EMBEDDING_REGISTRY",
    "FEATURE_EXTRACTOR_REGISTRY",
    "GUARDRAIL_REGISTRY",
    "LLM_REGISTRY",
    "LOADER_REGISTRY",
    "PREPROCESSOR_REGISTRY",
    "RERANKER_REGISTRY",
    "SEGMENTER_REGISTRY",
    "TERMINOLOGY_REGISTRY",
    "VECTORSTORE_REGISTRY",
    "OpenBTKError",
    "__version__",
    "get_registry",
]


def _detect_version() -> str:
    """Resolve the installed package version from distribution metadata.

    Falls back to a sentinel for a source tree that has never been installed.
    """
    try:
        return version("openbtk")
    except PackageNotFoundError:
        return "0.0.0.dev0+unknown"


__version__: str = _detect_version()
