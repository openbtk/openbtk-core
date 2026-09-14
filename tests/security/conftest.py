"""Shared adversarial fixtures for tests/security/.

These are release blockers (docs/07_TEST_CHARTER.md section 3.5): a failure
here stops a release regardless of schedule. Corresponds to the threats in
docs/06_SECURITY_COMPLIANCE.md section 1.

Scope note from M1 (closed as of M3, tasks 3.6/3.7): the doc's illustrative
template (docs/07_TEST_CHARTER.md section 3.5) references
run_full_pipeline(), labelled_phi_corpus, RunManifest.model_dump_json(), and
DeidReport. At M1 none of RunManifest, DeidReport or the pipeline executor
existed yet. All three now do -- see test_phi_in_run_manifest.py for the
test that gap was deferring to.
"""

from __future__ import annotations

import pytest

# Realistic-*looking* but entirely fabricated identifiers -- properly
# formatted enough that a naive "does this look like PHI" scan would flag
# them, which is exactly what makes them useful for this suite. None of
# these correspond to a real person; format only.
REALISTIC_PHI_STRINGS: list[str] = [
    "123-45-6789",  # SSN-shaped
    "MRN-00248193",  # a plausible medical record number
    "Jane Q. Patient",  # a plausible full name
    "DOB: 1958-03-14",  # a plausible date of birth
    "555-0142",  # a plausible phone extension
    "jane.patient@example-hospital.org",  # a plausible email
]


@pytest.fixture(params=REALISTIC_PHI_STRINGS)
def realistic_phi_string(request: pytest.FixtureRequest) -> str:
    """Parametrized fixture yielding one realistic-looking PHI string per
    test invocation, so every adversarial test in this suite is
    automatically run against all of them."""
    return str(request.param)
