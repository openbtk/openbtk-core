"""OpenBTKChatModel driven through real LangChain machinery (invoke, prompt
templates, output parsers, batch, stream)."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("langchain_core")

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from openbtk.core.errors import ProviderError
from openbtk.core.schemas import LLMResponse, Message
from openbtk.integrations.langchain import OpenBTKChatModel

from ._doubles import EchoLLM


def _model() -> tuple[OpenBTKChatModel, EchoLLM]:
    llm = EchoLLM()
    return OpenBTKChatModel(provider=llm), llm


class TestInvoke:
    def test_is_a_langchain_chat_model(self) -> None:
        assert isinstance(_model()[0], BaseChatModel)

    def test_string_input_becomes_a_user_message(self) -> None:
        model, llm = _model()
        out = model.invoke("hello")
        assert isinstance(out, AIMessage)
        assert out.content == "HELLO"
        assert [(m.role, m.content) for m in llm.seen] == [("user", "hello")]

    def test_roles_are_mapped_in_order(self) -> None:
        model, llm = _model()
        model.invoke(
            [
                SystemMessage(content="be brief"),
                HumanMessage(content="q1"),
                AIMessage(content="a1"),
                HumanMessage(content="q2"),
            ]
        )
        assert [m.role for m in llm.seen] == ["system", "user", "assistant", "user"]

    def test_token_usage_is_reported_as_langchain_usage_metadata(self) -> None:
        out = _model()[0].invoke("x")
        assert out.usage_metadata == {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
        }

    def test_a_response_without_usage_has_no_usage_metadata(self) -> None:
        class _NoUsage(EchoLLM):
            def chat(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
                return LLMResponse(text="ok")

        out = OpenBTKChatModel(provider=_NoUsage()).invoke("x")
        assert out.usage_metadata is None

    def test_stop_sequences_reach_the_provider(self) -> None:
        model, llm = _model()
        model.invoke("x", stop=["END"])
        assert llm.kwargs == {"stop": ["END"]}

    def test_no_stop_means_no_kwarg(self) -> None:
        model, llm = _model()
        model.invoke("x")
        assert llm.kwargs == {}

    def test_identifying_params_name_the_provider(self) -> None:
        model, _ = _model()
        assert model._llm_type == "openbtk"
        assert model._identifying_params == {"provider": "EchoLLM"}


class TestRefusals:
    def test_a_message_type_openbtk_cannot_carry_is_refused(self) -> None:
        with pytest.raises(ProviderError, match="ToolMessage"):
            _model()[0].invoke([ToolMessage(content="r", tool_call_id="1")])

    def test_multipart_content_is_refused_not_flattened(self) -> None:
        msg = HumanMessage(content=[{"type": "text", "text": "hi"}])
        with pytest.raises(ProviderError, match="plain-text"):
            _model()[0].invoke([msg])


class TestInLangChainPipelines:
    def test_prompt_template_to_model_to_parser(self) -> None:
        chain = (
            ChatPromptTemplate.from_messages([("human", "Summarise: {note}")])
            | _model()[0]
            | StrOutputParser()
        )
        assert chain.invoke({"note": "stable"}) == "SUMMARISE: STABLE"

    def test_batch(self) -> None:
        assert [m.content for m in _model()[0].batch(["a", "b"])] == ["A", "B"]

    def test_stream_falls_back_to_the_whole_response_as_one_chunk(self) -> None:
        chunks = list(_model()[0].stream("hi"))
        assert "".join(str(c.content) for c in chunks) == "HI"
