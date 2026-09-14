"""Root conftest: fixtures and collection hooks shared across every test
package.

docs/07_TEST_CHARTER.md section 3.5 uses ``labelled_phi_corpus`` as a
fixture name directly in its example test signatures
(``def test_no_phi_in_run_manifest(labelled_phi_corpus): ...``) -- defining
it here, at the tests/ root, makes it available to every test package
(security, accuracy, unit, contract) without each one needing its own copy
or its own import of ``tests.fixtures.labelled_phi_corpus`` (which does not
resolve the same way across packages -- see that module's own docstring
and tests/fixtures/test_labelled_phi_corpus.py's relative import for why).

Session-scoped: the corpus is deterministic (fixed seed) and its models are
frozen, so building it once per test session is both safe and much faster
than rebuilding it per test.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from fixtures.labelled_phi_corpus import LabelledPHICorpus, build_labelled_phi_corpus

if TYPE_CHECKING:
    from collections.abc import Sequence


@pytest.fixture(scope="session")
def labelled_phi_corpus() -> LabelledPHICorpus:
    return build_labelled_phi_corpus()


def pytest_collection_modifyitems(
    config: pytest.Config, items: Sequence[pytest.Item]
) -> None:
    """Activates the ``slow`` marker pyproject.toml declares but does not
    yet enforce ("requires model downloads (skipped unless
    OPENBTK_SLOW_TESTS=1)"): openbtk.deid.recognizers.ner's real
    detect()-calling tests are the first thing that needs it, and every
    future model-backed recognizer (task 2.9's LLM verifier included) gets
    the same behaviour for free.

    Deliberately an env var, not a CLI flag: CI and local "just run the
    fast suite" both want the same zero-configuration default, and a
    developer who wants the slow tests sets one variable rather than
    remembering a flag every invocation.
    """
    del config
    if os.environ.get("OPENBTK_SLOW_TESTS") == "1":
        return
    skip_slow = pytest.mark.skip(
        reason="requires model download; set OPENBTK_SLOW_TESTS=1"
    )
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
