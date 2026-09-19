"""The numbers in ``mkdocs/benchmarks.md`` must be the numbers the harness
produces now (CLAUDE.md rule 14). The published tables are compared verbatim
against a fresh run, so a recognizer change that moves a score cannot leave
the docs quietly claiming the old one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from openbtk.eval.deid import format_markdown

from .report import run

_DOC = Path(__file__).resolve().parents[2] / "mkdocs" / "benchmarks.md"


def _published(name: str) -> str:
    text = _DOC.read_text(encoding="utf-8")
    match = re.search(
        rf"<!-- benchmark:{name}:start -->\n(.*?)\n<!-- benchmark:{name}:end -->",
        text,
        re.DOTALL,
    )
    assert match is not None, f"benchmark:{name} markers missing from {_DOC.name}"
    return match.group(1).strip()


def _fresh(recognizers: list[str]) -> str:
    title = "Synthetic corpus, recognizers: " + " + ".join(recognizers)
    return format_markdown(run(recognizers), title=title).strip()


def test_rule_only_table_matches_a_fresh_run() -> None:
    assert _published("rule") == _fresh(["rule"])


@pytest.mark.slow
def test_ensemble_table_matches_a_fresh_run() -> None:
    assert _published("ner") == _fresh(["rule", "ner"])


def test_the_page_states_that_n2c2_was_not_run() -> None:
    """A guard against the page ever implying a real-corpus number."""
    text = _DOC.read_text(encoding="utf-8")
    assert "**Not run.**" in text
