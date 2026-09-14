# Contributing to OpenBTK

Thanks for considering a contribution. OpenBTK is early — milestones M1
through M3 (core framework, de-identification, clinical text + streaming
pipelines) are built and merged; the project is still finding its shape past
that. Please open an issue to discuss anything beyond a small fix before
starting substantial work, so effort isn't spent on something that doesn't
fit.

## Ground rules

- **PHI safety is absolute.** Never commit real patient data — not in code,
  not in a test fixture, not in an example, not in a commit message. Every
  fixture in this repository is synthetic (Faker-generated or hand-built) and
  must stay that way. If you're unsure whether something counts, ask before
  committing it.
- **"Done" means tests pass in CI, not "the file exists" or "it compiles."**
  A previous attempt at this project produced thousands of lines that were
  never actually executed; every rule in this file traces back to preventing
  that specific failure mode.
- **No claim without a benchmark.** An accuracy or performance number in a
  docstring, commit message, or PR description needs a reproducible test
  backing it up.

## Setting up a development environment

Requires Python 3.11+.

```bash
git clone https://github.com/openbtk/openbtk-core.git
cd openbtk-core
python -m venv .venv
source .venv/bin/activate       # .venv\Scripts\activate on Windows
pip install -e ".[dev,text]"
```

`dev` brings in pytest, mypy, ruff and friends; `text` brings in the clinical
text modality's optional dependencies (medspaCy, spaCy, pandas) so the full
test suite can actually run. Install additional extras (`ehr`, `retrieval`,
`llms`, `langchain`) as needed for the area you're working on, or `all` for
everything.

Install the pre-commit hooks so lint/type/security checks run before you
commit rather than after you push:

```bash
pre-commit install
```

## Running the test suite

```bash
pytest
```

This runs the default (fast) suite. Two categories are gated behind
environment variables and skipped otherwise:

```bash
OPENBTK_SLOW_TESTS=1 pytest      # tests needing a downloaded model (spaCy, HF)
OPENBTK_RUN_BENCHMARKS=1 pytest  # performance benchmarks (the 10M-note memory
                                  # benchmark takes on the order of 20-30 minutes)
```

Run tests **from outside the repository root** before considering something
done — a path-relative import can pass locally and fail once actually
installed. Move up one directory from your clone (`cd ..`) and run:

```bash
python -m pytest openbtk-core/tests --rootdir=openbtk-core
```

## Linting, formatting and type checking

```bash
ruff check .
ruff format --check .
mypy --strict
```

`mypy --strict` must report zero errors. `ruff check` and `ruff format
--check` must both be clean. CI runs all three; pre-commit runs the first two
(plus `detect-secrets` and a few project-specific guards) on every commit.

## Coding conventions

- One package name: `openbtk`, at `src/openbtk/`. Every internal import is
  absolute (`from openbtk.core... `), never relative, never `src.`.
- New schemas are Pydantic v2, `frozen=True`, `extra="forbid"`.
- New components register with a category registry
  (`@LOADER_REGISTRY.register("loader.<scope>.<name>")` and similar) rather
  than being imported directly by callers. Registry keys are permanent once
  merged — renaming one later needs `register_alias()` and a deprecation
  cycle, not a silent rename.
- Loaders and chunkers stream: `Iterator`, never `list`. A single record can
  produce thousands of chunks, and memory must stay `O(batch)`, not
  `O(corpus)`.
- Heavy optional dependencies (spaCy, medspaCy, pandas, transformers) are
  imported lazily, inside the method that needs them, via
  `openbtk.core._lazy.require(...)` — never at module scope. `import openbtk`
  itself must keep working with zero extras installed.
- Every public class and function gets a Google-style docstring with `Args`,
  `Returns`/`Yields`, `Raises`, and a runnable `Example` where practical.
  `pytest --doctest-modules` actually executes these.
- Every new base-class implementation needs to pass the shared contract
  suite in `tests/contract/` for that base class — there are no exemptions.
- Never log, print, or otherwise surface raw patient data. Use
  `openbtk.core.logging.get_logger`, never `structlog`/`logging` directly —
  the former's redaction processor is what keeps PHI out of logs by
  construction.

## Making a pull request

1. Fork the repository and create a branch off `main`.
2. Make your change, with tests that fail without it.
3. Run the full check list above.
4. Open a pull request describing what changed and why. Link the issue it
   addresses, if any.
5. A maintainer will review; expect some back-and-forth, especially on
   anything that touches `openbtk.deid` (PHI safety) or adds a new
   dependency.

## Reporting a security issue

Do **not** open a public issue for a security vulnerability, especially
anything involving a PHI leak path. See [SECURITY.md](SECURITY.md) for how to
report privately.

## Code of Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). Participation
in any project space implies agreement to abide by it.
