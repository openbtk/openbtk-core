"""Unit tests for openbtk.llms.base's re-exports.

base.py's whole content, after task 5.4 moved retry_with_backoff's real
definition to openbtk.core.retry (tests/unit/core/test_retry.py), is
re-exporting core.schemas/core.base/core.retry symbols for ergonomic
import -- verify they are the real, same objects, not shadow copies, the
same pattern already used for core.provenance's TokenUsage re-export
(tests/unit/core/test_provenance.py).
"""

from __future__ import annotations

from openbtk.llms.base import (
    BaseLLMProvider,
    LLMResponse,
    Message,
    TokenUsage,
    retry_with_backoff,
)


class TestReExports:
    def test_message_is_the_real_core_schemas_class(self) -> None:
        from openbtk.core.schemas import Message as CoreMessage

        assert Message is CoreMessage

    def test_llmresponse_is_the_real_core_schemas_class(self) -> None:
        from openbtk.core.schemas import LLMResponse as CoreLLMResponse

        assert LLMResponse is CoreLLMResponse

    def test_tokenusage_is_the_real_core_schemas_class(self) -> None:
        from openbtk.core.schemas import TokenUsage as CoreTokenUsage

        assert TokenUsage is CoreTokenUsage

    def test_basellmprovider_is_the_real_core_base_class(self) -> None:
        from openbtk.core.base import BaseLLMProvider as CoreBaseLLMProvider

        assert BaseLLMProvider is CoreBaseLLMProvider

    def test_retry_with_backoff_is_the_real_core_retry_function(self) -> None:
        from openbtk.core.retry import retry_with_backoff as core_retry_with_backoff

        assert retry_with_backoff is core_retry_with_backoff
