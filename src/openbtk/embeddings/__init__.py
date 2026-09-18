"""Embedding providers, biomedical and general.

Same registration-side-effect pattern as ``openbtk.llms``'s own
``__init__.py``: importing this package is what makes
``EMBEDDING_REGISTRY.create("embedding.general.huggingface")`` (etc.)
work without every caller separately importing the submodule. Neither
``huggingface.py`` nor ``openai.py`` pulls in a heavy optional dependency
by itself -- the real SDK/model import is lazy, inside a method, via
``openbtk.core._lazy.require``. ``presets.py`` has no registration side
effect of its own (it is a lookup table over ``huggingface.py``'s
provider, not a new component) but is imported here too for the same
"one import, everything available" reason.
"""

from __future__ import annotations

from openbtk.embeddings import huggingface, openai, presets

__all__ = ["huggingface", "openai", "presets"]
