"""One-command de-identification benchmark (roadmap task 8.1)::

    python -m openbtk.eval.deid_benchmark --dataset n2c2 --path /data/n2c2_2014 \
        --recognizers rule ner --json result.json --markdown result.md

Runs the real ``DeidEngine`` over a credentialed dataset the caller already
holds and scores it with the same code M2's CI accuracy gate uses
(``openbtk.eval.deid``). Documents are streamed one at a time; the report
holds counts and category names only, never document text.

**This command has not been run against the real n2c2/i2b2 corpus** -- that
data is Data-Use-Agreement-restricted and the environment this was built in
does not have it. It is exercised end to end against hand-built files in the
same format (``tests/unit/eval/test_deid_benchmark.py``). No i2b2/n2c2 number
is published anywhere in this repository; see ``mkdocs/benchmarks.md``.

Exit status: 0 on success, 2 if the dataset is unavailable or malformed, or a
requested recognizer's optional dependency is missing (message on stderr).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from openbtk.core.errors import OpenBTKError
from openbtk.core.registry import DATASET_REGISTRY
from openbtk.deid import DeidEngine, DeidMode
from openbtk.eval.deid import evaluate_deid, format_markdown, report_dict

if TYPE_CHECKING:
    from collections.abc import Sequence

_DATASET_KEYS = {"n2c2": "dataset.clinical_text.n2c2_deid"}


def _err(message: str) -> None:
    # Not print(): CLAUDE.md rule 10 reserves output for structlog, and a CLI's
    # own user-facing stream is the one deliberate exception, written directly.
    sys.stderr.write(message + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m openbtk.eval.deid_benchmark",
        description="Score DeidEngine against a credentialed de-id dataset.",
    )
    parser.add_argument("--dataset", required=True, choices=sorted(_DATASET_KEYS))
    parser.add_argument(
        "--path", required=True, help="Directory holding the dataset you obtained."
    )
    parser.add_argument(
        "--recognizers",
        nargs="+",
        default=["rule"],
        help='Recognizers to enable (default: "rule"; "ner" needs the text extra).',
    )
    parser.add_argument("--limit", type=int, help="Score only the first N documents.")
    parser.add_argument("--json", type=Path, help="Write the full report as JSON.")
    parser.add_argument("--markdown", type=Path, help="Write the report as Markdown.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        _err("--limit must be >= 1")
        return 2
    # Imported for its registration side effect (the dataset adapter).
    from openbtk.data import clinical_text as _clinical_text  # noqa: F401

    try:
        dataset = DATASET_REGISTRY.create(_DATASET_KEYS[args.dataset])
        documents = dataset.load(path=args.path)
        if args.limit is not None:
            documents = itertools.islice(documents, args.limit)
        engine = DeidEngine(mode=DeidMode.REDACT, recognizers=args.recognizers)
        result = evaluate_deid(engine, documents)
    except OpenBTKError as e:
        _err(f"error: {e}")
        return 2
    if result.n_documents == 0:
        _err("error: the dataset directory contains no *.xml documents.")
        return 2

    title = f"{dataset.name} -- recognizers: {', '.join(args.recognizers)}"
    markdown = format_markdown(result, title=title)
    if args.markdown:
        args.markdown.write_text(markdown, encoding="utf-8")
    if args.json:
        payload = {
            "dataset": dataset.name,
            "recognizers": list(args.recognizers),
            **report_dict(result),
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(markdown + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
