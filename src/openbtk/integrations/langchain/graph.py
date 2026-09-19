"""OpenBTK components as LangGraph nodes.

A LangGraph node is just a callable from the graph state to a partial state
update, so this module needs -- and imports -- neither ``langgraph`` nor
anything beyond the sibling ``runnables`` module. Compatibility with the real
library is verified by tests that build and run an actual ``StateGraph``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openbtk.core.errors import ConfigError
from openbtk.integrations.langchain.runnables import as_runnable

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from openbtk.core.base import Component
    from openbtk.deid.engine import DeidEngine


def as_langgraph_node(
    component: Component | DeidEngine,
    *,
    input_key: str,
    output_key: str,
    name: str | None = None,
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Wrap ``component`` as a LangGraph node.

    The node reads ``state[input_key]``, runs the component exactly as
    :func:`~openbtk.integrations.langchain.as_runnable` documents, and returns
    ``{output_key: result}`` -- a partial update LangGraph merges into state.

    Args:
        component: Any component ``as_runnable`` supports.
        input_key: State key holding the component's input.
        output_key: State key the result is written to.
        name: The node's ``__name__`` (LangGraph's default node name).
            Defaults to the component's class name.

    Raises:
        ConfigError: At call time, if ``input_key`` is missing from the state
            (the message names the key, never the state's contents).

    Example:
        >>> import contextlib, io
        >>> with contextlib.redirect_stdout(io.StringIO()):  # registry debug logs
        ...     from openbtk.guardrails.phi_leakage import PHILeakageGuardrail
        ...     node = as_langgraph_node(
        ...         PHILeakageGuardrail(), input_key="draft", output_key="check"
        ...     )
        ...     update = node({"draft": "Plan: rest."})
        >>> update["check"].passed
        True
    """
    runnable = as_runnable(component)

    def node(state: Mapping[str, Any]) -> dict[str, Any]:
        if input_key not in state:
            raise ConfigError(
                f"Graph state has no {input_key!r} key for node {node.__name__!r}.",
                context={"input_key": input_key},
            )
        return {output_key: runnable.invoke(state[input_key])}

    node.__name__ = name or type(component).__name__
    return node
