"""LLM providers. Biomedical models are configured presets of the HuggingFace
and endpoint providers, not bespoke classes.

``llms.base`` holds the shared building blocks (message/response/token-usage
re-exports plus :func:`~openbtk.llms.base.retry_with_backoff`) every
concrete provider is built from.

The four concrete providers (task 5.2) are imported here purely for their
``@LLM_REGISTRY.register(...)`` side effect, the same pattern
``data.clinical_text``'s ``__init__.py`` uses -- importing this package is
what makes ``LLM_REGISTRY.create("llm.general.openai")`` (etc.) work
without every caller separately importing ``openbtk.llms.openai``. None of
the four imports pulls in a heavy optional dependency by itself: every
provider's real SDK/model import is lazy, inside a method, via
``openbtk.core._lazy.require`` (``openai.py``, ``anthropic.py``,
``huggingface.py``) or is ``httpx`` (already core, ``openai_compatible.py``).
"""

from __future__ import annotations

from openbtk.llms import anthropic, huggingface, openai, openai_compatible

__all__ = ["anthropic", "huggingface", "openai", "openai_compatible"]
