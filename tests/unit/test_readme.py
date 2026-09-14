"""Executes every ```python code block in README.md (M4 task 4.4:
docs/10_ROADMAP.md's own "Remaining before the real v0.1.0" note --
"executing README code blocks in CI so a claim can't silently go stale").

v1's README advertised a quick-start importing a module that did not
exist. The only way this class of drift is actually prevented -- not just
discouraged -- is running the real code on every change, the same way a
docstring's ``Example:`` block is verified by ``--doctest-modules``
elsewhere in this project.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_README = _REPO_ROOT / "README.md"
_PYTHON_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _extract_python_blocks(markdown: str) -> list[str]:
    return _PYTHON_BLOCK_RE.findall(markdown)


def test_readme_has_at_least_one_python_block() -> None:
    """Sanity check that this test is not silently vacuous -- a README
    with no fenced ```python block at all would make every other check
    in this file pass trivially without verifying anything."""
    blocks = _extract_python_blocks(_README.read_text(encoding="utf-8"))
    assert blocks


def test_readme_python_blocks_execute_without_error() -> None:
    markdown = _README.read_text(encoding="utf-8")
    for i, block in enumerate(_extract_python_blocks(markdown)):
        namespace: dict[str, object] = {"__name__": "__readme_example__"}
        try:
            exec(compile(block, f"README.md:block[{i}]", "exec"), namespace)
        except Exception as e:  # pragma: no cover -- the whole point is to surface this
            raise AssertionError(
                f"README.md's python code block #{i} raised {type(e).__name__}: "
                f"{e}. A README example must actually run -- see this file's "
                "own module docstring for why."
            ) from e
