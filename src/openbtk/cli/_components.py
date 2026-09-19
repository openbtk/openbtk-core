"""Make every built-in component visible to the registries.

Registration is an import side effect, and OpenBTK deliberately does not import
every component package from ``import openbtk`` (that would slow every use of
the library). A command that lists, validates or runs components needs them
all registered, so the CLI imports them here. None of these packages needs an
optional dependency merely to be *defined* -- ``tests/packaging`` proves each
imports with zero extras -- so this cannot fail on a missing extra.
"""

from __future__ import annotations

import importlib

COMPONENT_PACKAGES = (
    "openbtk.data.clinical_text",
    "openbtk.data.ehr",
    "openbtk.embeddings",
    "openbtk.llms",
    "openbtk.retrieval",
    "openbtk.guardrails",
    "openbtk.terminology",
)


def load_components() -> None:
    """Import every built-in component package (plugins load on first
    registry access, as everywhere else)."""
    for name in COMPONENT_PACKAGES:
        importlib.import_module(name)
