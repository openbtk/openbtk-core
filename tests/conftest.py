"""Root conftest: fixtures shared across every test package.

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

import pytest

from fixtures.labelled_phi_corpus import LabelledPHICorpus, build_labelled_phi_corpus


@pytest.fixture(scope="session")
def labelled_phi_corpus() -> LabelledPHICorpus:
    return build_labelled_phi_corpus()
