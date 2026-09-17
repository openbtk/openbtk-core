"""LLM provider building blocks: messages, responses, retry/backoff, token
accounting.

``Message``, ``LLMResponse``, ``TokenUsage`` and ``BaseLLMProvider`` are
defined in ``core.schemas``/``core.base`` -- every modality-neutral schema
and abstract base lives there, not per-modality (docs/04_API_DESIGN.md
section 3). They are re-exported here so a concrete provider (task 5.2)
needs one import line -- ``from openbtk.llms.base import (BaseLLMProvider,
LLMResponse, Message, TokenUsage, retry_with_backoff)`` -- instead of
reaching into ``core`` directly.

``retry_with_backoff`` was originally defined here (task 5.2); moved to
``openbtk.core.retry`` at task 5.4 once ``embeddings/openai.py`` needed
the exact same rate-limited-then-retry shape and had nothing LLM-specific
to justify importing it from this, an unrelated same-level provider
category. Re-exported here for backward compatibility with the name it
was under, and so every existing provider's
``from openbtk.llms.base import retry_with_backoff`` keeps working
unchanged.
"""

from __future__ import annotations

from openbtk.core.base import BaseLLMProvider
from openbtk.core.retry import retry_with_backoff
from openbtk.core.schemas import LLMResponse, Message, TokenUsage

__all__ = [
    "BaseLLMProvider",
    "LLMResponse",
    "Message",
    "TokenUsage",
    "retry_with_backoff",
]
