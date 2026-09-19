"""Every python block on mkdocs/langchain.md is executed, in order, in one
namespace -- the page may not show code that does not run (CLAUDE.md: "every
code block tested")."""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")

_PAGE = Path(__file__).resolve().parents[3] / "mkdocs" / "langchain.md"


def _blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", _PAGE.read_text("utf-8"), re.DOTALL)


def test_the_page_has_code_blocks() -> None:
    assert len(_blocks()) >= 4


def test_every_code_block_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("langgraph")
    # The blocks run as a real module, so what they define (a TypedDict whose
    # annotations LangGraph resolves by module lookup) behaves as it would
    # for a reader who pastes them into a file.
    module = types.ModuleType("langchain_docs_page")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    for i, block in enumerate(_blocks()):
        exec(compile(block, f"{_PAGE.name}[block {i}]", "exec"), module.__dict__)
