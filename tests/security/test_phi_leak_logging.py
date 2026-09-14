"""Release blocker: realistic-looking PHI must never survive into a log
line, at any deny-listed field, using the same redaction machinery
tests/unit/core/test_logging.py already verifies unit-by-unit.

That file proves the mechanism works (fixed sentinel values, one behaviour
per test). This file is the release-gate framing docs/07_TEST_CHARTER.md
section 3.5 describes: fuzz every realistic-looking identifier in
conftest.py's REALISTIC_PHI_STRINGS through every deny-listed field, in one
sweep, so a release is blocked on ANY single regression rather than relying
on remembering to keep the two files in sync by hand.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from typing import Any

from openbtk.core.logging import get_logger

_DENY_LIST_KEYS = [
    "raw_text",
    "text",
    "note",
    "content",
    "patient_name",
    "name",
    "mrn",
    "ssn",
    "dob",
    "address",
    "phone",
    "email",
]


def _capture(log_call: Any) -> dict[str, Any]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        log_call()
    return dict(json.loads(buf.getvalue().strip()))


class TestNoRealisticPhiSurvivesLogging:
    def test_no_deny_listed_field_carries_realistic_phi_through(
        self, realistic_phi_string: str
    ) -> None:
        log = get_logger("security.release_gate")
        for key in _DENY_LIST_KEYS:
            out = _capture(
                lambda k=key: log.info(
                    "security.release_gate.probe", **{k: realistic_phi_string}
                )
            )
            assert key not in out, (
                f"field {key!r} survived redaction carrying "
                f"{realistic_phi_string!r} -- this is a release blocker."
            )
            # Belt and braces: the value must not appear ANYWHERE in the
            # emitted line, even under an unexpected key, in case a future
            # change to the redaction processor renames rather than drops.
            assert realistic_phi_string not in json.dumps(out), (
                f"{realistic_phi_string!r} leaked into the log line under "
                f"a different key than {key!r} -- this is a release blocker."
            )

    def test_realistic_phi_embedded_in_an_unlisted_field_is_not_specially_caught(
        self, realistic_phi_string: str
    ) -> None:
        """Documents a real, known limitation rather than pretending
        coverage that doesn't exist: the redaction processor is a
        deny-list on FIELD NAMES (core/logging.py), not a content scanner.
        A realistic PHI-shaped string logged under an arbitrary, non-deny-
        listed key (e.g. "detail") is NOT redacted -- callers are
        responsible for never passing patient content under an unlisted
        key. This test exists so that if a content-scanning layer is ever
        added, this file has a matching test to update rather than one that
        silently under-tests it forever."""
        log = get_logger("security.release_gate")
        out = _capture(
            lambda: log.info("security.release_gate.probe", detail=realistic_phi_string)
        )
        assert out["detail"] == realistic_phi_string
