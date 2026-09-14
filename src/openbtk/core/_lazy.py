"""Optional-dependency import helper.

This is what lets ``import openbtk.data.clinical_text`` succeed with zero
extras installed, and only fail -- with an actionable message -- when a
component that actually needs (say) scispaCy is constructed.

Rule (docs/09_CODING_STANDARDS.md section 4): heavy imports live inside
methods or ``__init__``, resolved via :func:`require`, never as a bare
module-level ``import``. A bare ``import torch`` at module scope breaks the
zero-extras install for every caller of that module, not just the caller who
needed torch.

This module is private (leading underscore) -- it is plumbing for component
authors, not part of the public API in docs/04_API_DESIGN.md.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from openbtk.core.errors import MissingDependencyError

if TYPE_CHECKING:
    from types import ModuleType

# Process-wide cache so repeated require() calls for the same module (e.g. one
# per component instantiation) do not re-run importlib's module search on
# every call. importlib.import_module already caches in sys.modules, but this
# avoids the dict lookup and attribute chain on the hot path.
_CACHE: dict[str, ModuleType] = {}


def require(module: str, *, extra: str) -> ModuleType:
    """Import an optional dependency, or raise an actionable error.

    Args:
        module: Dotted module path to import, e.g. ``"scispacy"`` or
            ``"fhir.resources"``.
        extra: The optional-dependency group name from ``pyproject.toml``
            that provides this module, e.g. ``"text"``. Used to build the
            install instruction in the error message.

    Returns:
        The imported module.

    Raises:
        MissingDependencyError: If the module cannot be imported. The message
            names the exact ``pip install`` command to run; ``.context``
            carries ``module`` and ``extra`` for programmatic handling.

    Example:
        >>> mod = require("json", extra="never-missing")  # stdlib, always present
        >>> mod.__name__
        'json'
    """
    cached = _CACHE.get(module)
    if cached is not None:
        return cached

    try:
        imported = importlib.import_module(module)
    except ImportError as e:
        raise MissingDependencyError(
            f"'{module}' is required for this component but is not installed.\n"
            f"Install it with:  pip install 'openbtk[{extra}]'",
            context={"module": module, "extra": extra},
        ) from e

    _CACHE[module] = imported
    return imported
