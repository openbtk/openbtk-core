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

__all__ = ["__version__"]


def _detect_version() -> str:
    """Resolve the installed package version from distribution metadata.

    Falls back to a sentinel for a source tree that has never been installed.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("openbtk")
    except PackageNotFoundError:
        return "0.0.0.dev0+unknown"


__version__: str = _detect_version()
