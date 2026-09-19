"""An OpenBTK LLM provider as a ``langchain_core`` chat model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict

from openbtk.core.base import BaseLLMProvider  # noqa: TC001 -- pydantic field type
from openbtk.core.errors import ProviderError
from openbtk.core.schemas import Message

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForLLMRun


def _to_openbtk(messages: list[BaseMessage]) -> list[Message]:
    converted: list[Message] = []
    role: Literal["system", "user", "assistant"]
    for m in messages:
        if isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage):
            role = "assistant"
        elif isinstance(m, SystemMessage):
            role = "system"
        else:
            raise ProviderError(
                f"OpenBTKChatModel cannot send a {type(m).__name__}: OpenBTK "
                "providers take only system/user/assistant messages.",
                context={"message_type": type(m).__name__},
            )
        if not isinstance(m.content, str):
            raise ProviderError(
                "OpenBTKChatModel accepts only plain-text message content, not "
                "multi-part content blocks.",
                context={"message_type": type(m).__name__},
            )
        converted.append(Message(role=role, content=m.content))
    return converted


class OpenBTKChatModel(BaseChatModel):
    """Expose a ``BaseLLMProvider`` as a LangChain chat model.

    Only system, user and assistant messages with plain-text content are
    supported -- anything else raises ``ProviderError`` rather than being
    flattened into a lossy prompt. Streaming is not overridden: OpenBTK's
    ``BaseLLMProvider.stream`` takes a prompt string, not a message list, so
    ``.stream()`` here falls back to LangChain's default of yielding the
    complete response as a single chunk.

    Args:
        provider: The provider to wrap.

    Example:
        >>> from openbtk.core.base import BaseLLMProvider
        >>> from openbtk.core.schemas import LLMResponse
        >>> class Echo(BaseLLMProvider):
        ...     sends_data_offsite = False
        ...     def generate(self, prompt, **kw): return LLMResponse(text=prompt)
        ...     def stream(self, prompt, **kw): yield prompt
        ...     def chat(self, messages, **kw):
        ...         return LLMResponse(text=messages[-1].content.upper())
        >>> OpenBTKChatModel(provider=Echo()).invoke("hi").content
        'HI'
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: BaseLLMProvider

    @property
    def _llm_type(self) -> str:
        return "openbtk"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"provider": self.provider.registry_key or type(self.provider).__name__}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> ChatResult:
        if stop:
            kwargs["stop"] = stop
        response = self.provider.chat(_to_openbtk(messages), **kwargs)
        usage: UsageMetadata | None = None
        if response.usage is not None:
            usage = UsageMetadata(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        message = AIMessage(content=response.text, usage_metadata=usage)
        return ChatResult(generations=[ChatGeneration(message=message)])
