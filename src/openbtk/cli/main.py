"""Argument parsing and dispatch for the ``openbtk`` command.

Built on ``argparse`` (standard library), not Typer as docs/03_ARCHITECTURE.md
first sketched: the CLI has six small commands, and a command-line framework
would be a seventh runtime dependency for a core that is held at six (an ADR
would be needed). ``argparse`` also means ``openbtk`` works from a plain
``pip install openbtk`` with no extra. The commands are thin handlers, so
swapping the framework later would touch only this file.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from typing import TYPE_CHECKING

from openbtk.cli import commands
from openbtk.core.errors import (
    ConfigError,
    MissingDependencyError,
    OpenBTKError,
    PolicyError,
    RegistryError,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    Handler = Callable[[argparse.Namespace], tuple[int, str]]

# Errors that mean "the invocation or config is wrong", as opposed to "it ran
# and something failed".
_USAGE_ERRORS = (ConfigError, RegistryError, MissingDependencyError, PolicyError)


def build_parser() -> argparse.ArgumentParser:
    """The ``openbtk`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="openbtk",
        description="De-identified, auditable, model-ready biomedical data pipelines.",
    )
    parser.add_argument("--version", action="store_true", help="print the version")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    def add(name: str, help_: str, handler: Handler) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_, description=help_)
        p.set_defaults(handler=handler)
        return p

    p = add("list", "list registered components", commands.cmd_list)
    p.add_argument("category", nargs="?", help="a registry category, e.g. loader")
    p.add_argument("--json", action="store_true", help="machine-readable output")

    p = add(
        "validate", "check a pipeline config without running it", commands.cmd_validate
    )
    p.add_argument("config", help="path to a pipeline YAML")
    p.add_argument("--json", action="store_true", help="machine-readable output")

    p = add("run", "run a pipeline and write its manifest", commands.cmd_run)
    p.add_argument("config", help="path to a pipeline YAML")
    p.add_argument(
        "--manifest",
        help="where to write the manifest (default: the config's manifest_dir)",
    )
    p.add_argument("--json", action="store_true", help="print the manifest as JSON")
    p.add_argument(
        "--checkpoint",
        help=(
            "resume from this file if it exists, and periodically save how far each "
            "loader has read (FR-L-05); deleted on a successful run"
        ),
    )
    p.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="records between checkpoint saves (default: 1000)",
    )

    p = add(
        "deid", "de-identify a directory of notes or a .jsonl file", commands.cmd_deid
    )
    p.add_argument("input", help="a directory of .txt notes, or a .jsonl file")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument(
        "--mode",
        default="redact",
        choices=["redact", "tag", "hash", "surrogate", "date_shift"],
        help="how detected spans are transformed (default: redact)",
    )
    p.add_argument(
        "--recognizers",
        default="rule",
        help="comma-separated recognizers, e.g. rule,ner (default: rule)",
    )
    p.add_argument("--force", action="store_true", help="write into a non-empty --out")

    p = add("replay", "re-run a recorded manifest and compare", commands.cmd_replay)
    p.add_argument("manifest", help="path to a run manifest JSON")
    p.add_argument("--new-manifest", help="where to write the replay's manifest")
    p.add_argument("--json", action="store_true", help="machine-readable output")

    p = add(
        "doctor",
        "diagnose installed extras, models and credentials",
        commands.cmd_doctor,
    )
    p.add_argument(
        "--require", help="comma-separated extras that must be ready, e.g. text,ehr"
    )
    p.add_argument("--json", action="store_true", help="machine-readable output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Returns the process exit code (see ``commands``)."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:  # argparse exits on --help (0) and usage errors (2)
        return e.code if isinstance(e.code, int) else 2

    real_stdout = sys.stdout
    if args.version:
        import openbtk

        real_stdout.write(f"openbtk {openbtk.__version__}\n")
        return 0
    if not getattr(args, "handler", None):
        parser.print_usage(sys.stderr)
        return 2

    code, text = 2, ""
    # The library logs to stdout; route it to stderr for the duration so the
    # command's own output is the only thing on stdout.
    with contextlib.redirect_stdout(sys.stderr):
        try:
            code, text = args.handler(args)
        except OpenBTKError as e:
            # OpenBTK errors carry identifiers and messages, never PHI.
            sys.stderr.write(f"error: {e}\n")
            code = 2 if isinstance(e, _USAGE_ERRORS) else 1
    real_stdout.write(text)
    return code
