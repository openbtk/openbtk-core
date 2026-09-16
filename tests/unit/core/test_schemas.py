"""Unit tests for openbtk.core.schemas.

Most schemas here are already exercised via the contract suite and their
own doctests; this file covers what those don't: ``TokenUsage``'s
accumulation behaviour (moved here from test_provenance.py in M5 task
5.1, alongside the class itself -- see core.provenance's own module
docstring for why) and ``LLMResponse.usage``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from openbtk.core.schemas import LLMResponse, TokenUsage


class TestTokenUsage:
    def test_all_fields_default_to_zero(self) -> None:
        usage = TokenUsage()
        assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (
            0,
            0,
            0,
        )

    def test_rejects_negative_tokens(self) -> None:
        with pytest.raises(ValidationError):
            TokenUsage(prompt_tokens=-1)

    def test_is_frozen(self) -> None:
        usage = TokenUsage(total_tokens=5)
        with pytest.raises(ValidationError):
            usage.total_tokens = 10  # type: ignore[misc]

    def test_add_sums_every_field_independently(self) -> None:
        a = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        b = TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5)
        combined = a + b
        assert (combined.prompt_tokens, combined.completion_tokens) == (13, 7)
        assert combined.total_tokens == 20

    def test_add_returns_a_new_instance_not_a_mutation(self) -> None:
        a = TokenUsage(total_tokens=1)
        b = TokenUsage(total_tokens=2)
        combined = a + b
        assert a.total_tokens == 1  # unchanged
        assert b.total_tokens == 2  # unchanged
        assert combined is not a
        assert combined is not b

    def test_add_is_associative_for_accumulating_across_many_calls(self) -> None:
        """The real use case: summing N LLM calls' usage into one running
        total for a RunManifest, regardless of grouping order."""
        calls = [TokenUsage(total_tokens=i) for i in (1, 2, 3, 4)]
        left_to_right = calls[0] + calls[1] + calls[2] + calls[3]
        grouped = (calls[0] + calls[1]) + (calls[2] + calls[3])
        assert left_to_right.total_tokens == grouped.total_tokens == 10

    def test_add_rejects_a_non_tokenusage_operand(self) -> None:
        with pytest.raises(TypeError):
            TokenUsage(total_tokens=1) + 1  # type: ignore[operator]


class TestLLMResponse:
    def test_usage_defaults_to_none(self) -> None:
        assert LLMResponse(text="hello").usage is None

    def test_usage_round_trips(self) -> None:
        response = LLMResponse(text="hello", usage=TokenUsage(total_tokens=42))
        assert response.usage is not None
        assert response.usage.total_tokens == 42

    def test_is_frozen(self) -> None:
        response = LLMResponse(text="hello")
        with pytest.raises(ValidationError):
            response.text = "changed"  # type: ignore[misc]
