"""Print the de-identification benchmark table published in
``mkdocs/benchmarks.md`` (CLAUDE.md rule 14: no number without a harness).

Run from the repository root::

    python tests/accuracy/report.py                     # rule recognizer only
    python tests/accuracy/report.py --recognizers rule,ner   # needs the [text] extra

The corpus is the seeded synthetic ``labelled_phi_corpus`` -- Faker and
templates, no real patient data. The scoring code is ``openbtk.eval.deid``,
the same implementation the accuracy gate and the credentialed n2c2 harness
use, so the table below and the CI gate cannot disagree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openbtk.deid.engine import DeidEngine
from openbtk.eval.deid import EvaluationResult, evaluate_deid, format_markdown

_TESTS_DIR = Path(__file__).resolve().parent.parent


def run(recognizers: list[str]) -> EvaluationResult:
    """Score ``DeidEngine`` with ``recognizers`` on the seeded synthetic corpus."""
    if str(_TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(_TESTS_DIR))
    from fixtures.labelled_phi_corpus import build_labelled_phi_corpus

    corpus = build_labelled_phi_corpus()
    return evaluate_deid(DeidEngine(recognizers=recognizers), corpus.documents)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--recognizers", default="rule")
    args = parser.parse_args(argv)
    recognizers = [r for r in args.recognizers.split(",") if r]
    title = "Synthetic corpus, recognizers: " + " + ".join(recognizers)
    sys.stdout.write(format_markdown(run(recognizers), title=title))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
