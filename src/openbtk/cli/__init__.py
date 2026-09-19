"""The ``openbtk`` command line (``python -m openbtk`` or the ``openbtk`` script).

``main`` imports the implementation on first call, inside a stdout-to-stderr
redirect: component packages log their registration to stdout when imported,
and a script parsing ``openbtk list --json`` must find nothing but the JSON
there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return its exit code (see ``openbtk.cli.commands``)."""
    import contextlib
    import sys

    with contextlib.redirect_stdout(sys.stderr):
        from openbtk.cli.main import main as run

    return run(argv)
