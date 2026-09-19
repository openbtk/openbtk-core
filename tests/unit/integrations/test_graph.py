"""as_langgraph_node: the node contract, plus a real LangGraph StateGraph run
when langgraph is installed."""

from __future__ import annotations

from typing import Any, TypedDict

import pytest

pytest.importorskip("langchain_core")

from openbtk.core.errors import ConfigError
from openbtk.guardrails.phi_leakage import PHILeakageGuardrail
from openbtk.integrations.langchain import as_langgraph_node

from ._doubles import EchoLLM


class TestNodeContract:
    def test_reads_the_input_key_and_returns_a_partial_update(self) -> None:
        node = as_langgraph_node(EchoLLM(), input_key="prompt", output_key="answer")
        assert node({"prompt": "hi", "other": 1}) == {"answer": "HI"}

    def test_default_name_is_the_component_class(self) -> None:
        node = as_langgraph_node(EchoLLM(), input_key="a", output_key="b")
        assert node.__name__ == "EchoLLM"

    def test_name_can_be_overridden(self) -> None:
        node = as_langgraph_node(EchoLLM(), input_key="a", output_key="b", name="llm")
        assert node.__name__ == "llm"

    def test_a_missing_input_key_names_the_key_not_the_state(self) -> None:
        node = as_langgraph_node(EchoLLM(), input_key="prompt", output_key="answer")
        with pytest.raises(ConfigError, match="'prompt'") as exc:
            node({"secret": "PATIENT TEXT"})
        assert "PATIENT TEXT" not in str(exc.value)

    def test_unsupported_components_fail_at_wrapping_time(self) -> None:
        with pytest.raises(ConfigError):
            as_langgraph_node(object(), input_key="a", output_key="b")  # type: ignore[arg-type]


class _State(TypedDict, total=False):
    draft: str
    check: Any
    final: str


class TestInARealStateGraph:
    def test_nodes_run_in_a_compiled_langgraph_workflow(self) -> None:
        graph_mod = pytest.importorskip("langgraph.graph")
        graph = graph_mod.StateGraph(_State)
        graph.add_node(
            "shout",
            as_langgraph_node(EchoLLM(), input_key="draft", output_key="final"),
        )
        graph.add_node(
            "phi_check",
            as_langgraph_node(
                PHILeakageGuardrail(), input_key="final", output_key="check"
            ),
        )
        graph.add_edge(graph_mod.START, "shout")
        graph.add_edge("shout", "phi_check")
        graph.add_edge("phi_check", graph_mod.END)

        result = graph.compile().invoke({"draft": "plan: rest"})

        assert result["final"] == "PLAN: REST"
        assert result["check"].passed is True
        assert result["draft"] == "plan: rest"  # untouched keys are preserved
