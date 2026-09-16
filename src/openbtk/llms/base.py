"""LLM provider building blocks: messages, responses, retry/backoff, token
accounting.

``Message``, ``LLMResponse``, ``TokenUsage`` and ``BaseLLMProvider`` are
defined in ``core.schemas``/``core.base`` -- every modality-neutral schema
and abstract base lives there, not per-modality (docs/04_API_DESIGN.md
section 3). They are re-exported here so a concrete provider (task 5.2)
needs one import line -- ``from openbtk.llms.base import (BaseLLMProvider,
LLMResponse, Message, TokenUsage, retry_with_backoff)`` -- instead of
reaching into ``core`` directly.

:func:`retry_with_backoff` is the one genuinely new piece this task adds:
no concrete provider exists yet, but every one of them (OpenAI, Anthropic,
HuggingFace-local, OpenAI-compatible) will hit the same
rate-limited-then-retry shape, so it belongs here once, not duplicated
per provider.
"""

from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING, TypeVar

from openbtk.core.base import BaseLLMProvider
from openbtk.core.errors import RateLimitError
from openbtk.core.logging import get_logger
from openbtk.core.schemas import LLMResponse, Message, TokenUsage

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "BaseLLMProvider",
    "LLMResponse",
    "Message",
    "TokenUsage",
    "retry_with_backoff",
]

log = get_logger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    func: Callable[[], T],
    *,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 0.1,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``func``, retrying on :class:`RateLimitError` with exponential backoff.

    Args:
        func: A zero-argument callable to invoke, e.g.
            ``lambda: client.chat.completions.create(...)``.
        max_attempts: Total attempts before giving up. Must be >= 1; the
            final attempt's :class:`RateLimitError` propagates unretried.
        base_delay: Delay in seconds before the first retry.
        max_delay: Upper bound on the computed delay, before jitter is applied.
        jitter: Fraction of the computed delay to randomise by (+/-), so
            many callers backing off together do not retry in lockstep --
            a real thundering-herd risk against a shared rate limit, not a
            hypothetical one.
        sleep: Injectable in place of ``time.sleep`` -- tests pass a no-op
            so a multi-attempt backoff runs in milliseconds, not tens of
            seconds of real wall-clock delay.

    Returns:
        Whatever ``func()`` returns, from whichever attempt first succeeds.

    Raises:
        RateLimitError: If every attempt is exhausted still rate-limited.
        ValueError: If ``max_attempts`` is less than 1.

    Example:
        >>> retry_with_backoff(lambda: "ok")
        'ok'

    The retry-on-``RateLimitError`` path is not shown here since a
    successful retry logs a ``llm.rate_limited_retrying`` warning
    (docs/09_CODING_STANDARDS.md's structlog rule) whose timestamp would
    make this doctest non-deterministic; see
    ``tests/unit/llms/test_base.py`` for that behaviour, exercised directly.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}.")

    for attempt in range(max_attempts):
        try:
            return func()
        except RateLimitError:
            if attempt == max_attempts - 1:
                raise
            delay = min(base_delay * (2**attempt), max_delay)
            delay = max(delay + random.uniform(-jitter * delay, jitter * delay), 0.0)
            log.warning(
                "llm.rate_limited_retrying",
                attempt=attempt + 1,
                max_attempts=max_attempts,
                delay_seconds=round(delay, 3),
            )
            sleep(delay)

    # Unreachable: max_attempts >= 1 (checked above) guarantees at least one
    # loop iteration, and every iteration either returns or raises -- the
    # last one unconditionally. Satisfies mypy's "missing return" check
    # without asserting something false; see executor.py's own
    # defensive-backstop pragma for the same pattern.
    raise RuntimeError("unreachable")  # pragma: no cover
