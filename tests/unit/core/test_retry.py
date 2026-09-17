"""Unit tests for openbtk.core.retry.retry_with_backoff.

Moved here from tests/unit/llms/test_base.py at task 5.4, alongside the
function itself -- see core.retry's own module docstring for why
(embeddings/openai.py needed the identical rate-limited-then-retry shape,
with nothing LLM-specific to justify it living under llms/). No concrete
provider is exercised here either way -- tested directly against injected
functions, with an injectable ``sleep`` so these tests never wait in real
wall-clock time.
"""

from __future__ import annotations

import pytest

from openbtk.core.errors import ProviderError, RateLimitError
from openbtk.core.retry import retry_with_backoff


class _RecordingSleep:
    """A ``sleep`` stand-in that records delays instead of waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def test_returns_the_result_on_first_success_without_sleeping() -> None:
    sleeper = _RecordingSleep()
    result = retry_with_backoff(lambda: "ok", sleep=sleeper)
    assert result == "ok"
    assert sleeper.delays == []


def test_retries_on_rate_limit_error_and_eventually_succeeds() -> None:
    calls = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise RateLimitError("try again")
        return "ok"

    result = retry_with_backoff(flaky, sleep=_RecordingSleep())
    assert result == "ok"
    assert len(calls) == 3


def test_sleeps_between_each_retry_but_not_after_the_final_attempt() -> None:
    calls = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise RateLimitError("try again")
        return "ok"

    sleeper = _RecordingSleep()
    retry_with_backoff(flaky, sleep=sleeper)
    assert len(sleeper.delays) == 2  # two retries before the third, successful call


def test_delay_grows_exponentially_before_jitter() -> None:
    def always_rate_limited() -> str:
        raise RateLimitError("nope")

    sleeper = _RecordingSleep()
    with pytest.raises(RateLimitError):
        retry_with_backoff(
            always_rate_limited,
            max_attempts=4,
            base_delay=1.0,
            jitter=0.0,
            sleep=sleeper,
        )
    assert sleeper.delays == [1.0, 2.0, 4.0]


def test_delay_is_capped_at_max_delay() -> None:
    def always_rate_limited() -> str:
        raise RateLimitError("nope")

    sleeper = _RecordingSleep()
    with pytest.raises(RateLimitError):
        retry_with_backoff(
            always_rate_limited,
            max_attempts=5,
            base_delay=10.0,
            max_delay=15.0,
            jitter=0.0,
            sleep=sleeper,
        )
    assert all(delay <= 15.0 for delay in sleeper.delays)


def test_raises_the_final_rate_limit_error_once_attempts_are_exhausted() -> None:
    def always_rate_limited() -> str:
        raise RateLimitError("still limited")

    with pytest.raises(RateLimitError, match="still limited"):
        retry_with_backoff(always_rate_limited, max_attempts=3, sleep=_RecordingSleep())


def test_non_rate_limit_errors_propagate_immediately_without_retry() -> None:
    calls = []

    def broken() -> str:
        calls.append(1)
        raise ProviderError("auth failed, not a rate limit")

    with pytest.raises(ProviderError):
        retry_with_backoff(broken, sleep=_RecordingSleep())
    assert len(calls) == 1  # never retried


def test_rejects_max_attempts_below_one() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        retry_with_backoff(lambda: "unreachable", max_attempts=0)


def test_single_attempt_never_retries() -> None:
    calls = []

    def always_rate_limited() -> str:
        calls.append(1)
        raise RateLimitError("nope")

    with pytest.raises(RateLimitError):
        retry_with_backoff(always_rate_limited, max_attempts=1, sleep=_RecordingSleep())
    assert len(calls) == 1
