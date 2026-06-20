"""
openbtk plugin discovery via Python entry points.

Third-party packages that provide openbtk providers/components declare
them in their pyproject.toml:

    [project.entry-points."openbtk.providers"]
    my_embedder = "my_package.embeddings:register"

The value must be a callable that imports the component module (triggering
the @REGISTRY.register() decorators to execute).
"""
from __future__ import annotations

import importlib
import importlib.metadata
import structlog

from openbtk.core.errors import PluginError

log = structlog.get_logger(__name__)

_ENTRY_POINT_GROUP = "openbtk.providers"
_loaded = False


def load_plugins() -> list[str]:
    """Discover and load all registered openbtk provider plugins.

    Should be called once at application startup (or lazily on first registry
    access). Idempotent — subsequent calls are no-ops.

    Returns:
        List of successfully loaded plugin names.

    Raises:
        PluginError: If a plugin module fails to import (non-fatal by default;
            the error is logged and loading continues).
    """
    global _loaded
    if _loaded:
        return []

    loaded_names: list[str] = []
    try:
        eps = importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP)
    except Exception as e:
        log.warning("plugins.discovery_failed", error=str(e))
        _loaded = True
        return []

    for ep in eps:
        try:
            fn = ep.load()
            if callable(fn):
                fn()
            loaded_names.append(ep.name)
            log.info("plugins.loaded", plugin=ep.name, value=ep.value)
        except Exception as e:
            log.error(
                "plugins.load_failed",
                plugin=ep.name,
                value=ep.value,
                error=str(e),
            )
            # Non-fatal — other plugins continue loading

    _loaded = True
    log.info("plugins.discovery_complete", n_loaded=len(loaded_names))
    return loaded_names


def load_builtin_modules() -> None:
    """Import all built-in modality modules to trigger their registrations.

    Called automatically when a registry is first accessed. This ensures
    built-in components are available without requiring explicit imports.
    """
    _builtin_modules = [
        # Core providers — always load
        "openbtk.embeddings",
        "openbtk.llms",
        "openbtk.guardrails",
        "openbtk.retrieval",
        # Modality modules — these auto-register on import
        # (only if dependencies are installed, else skipped gracefully)
    ]

    _optional_modality_modules = [
        "openbtk.data.clinical_text",
        "openbtk.data.ehr_emr",
        "openbtk.data.imaging",
        "openbtk.data.biosignals",
        "openbtk.data.genomics",
        "openbtk.data.video",
        "openbtk.data.audio",
    ]

    for module_path in _builtin_modules:
        try:
            importlib.import_module(module_path)
        except ImportError as e:
            log.debug("plugins.builtin_skip", module=module_path, reason=str(e))

    for module_path in _optional_modality_modules:
        try:
            importlib.import_module(module_path)
            log.debug("plugins.modality_loaded", module=module_path)
        except ImportError as e:
            log.debug(
                "plugins.modality_skip",
                module=module_path,
                reason=str(e),
                hint="Install the matching optional extra, e.g. pip install openbtk[imaging]",
            )
