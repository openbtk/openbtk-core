"""Unit tests for openbtk.core.logging (PHI redaction, allow_phi toggling).

Converted from a throwaway adversarial probe run during development. That
probe is what actually caught the get_logger(name) binding bug (name was
never used) and confirmed the order-independence property below -- this
file makes those same checks permanent.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

import pytest

from openbtk.core.logging import configure_logging, get_logger

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


@pytest.fixture(autouse=True)
def _restore_safe_default() -> Iterator[None]:
    """allow_phi is a process-wide switch -- every test must leave it in
    the safe default state, regardless of what it did during the test."""
    yield
    configure_logging(allow_phi=False)


def _capture(log_call: Any) -> dict[str, Any]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        log_call()
    return dict(json.loads(buf.getvalue().strip()))


class TestRedactionDefault:
    @pytest.mark.parametrize("key", _DENY_LIST_KEYS)
    def test_every_deny_list_key_is_dropped_entirely(self, key: str) -> None:
        log = get_logger("test.module")
        out = _capture(lambda: log.info("test.event", **{key: "SENSITIVE-VALUE"}))
        assert key not in out

    def test_unlisted_fields_pass_through_unmodified(self) -> None:
        log = get_logger("test.module")
        out = _capture(lambda: log.info("test.event", safe_field=42))
        assert out["safe_field"] == 42

    def test_record_id_and_patient_id_are_hashed_not_raw(self) -> None:
        log = get_logger("test.module")
        out = _capture(
            lambda: log.info(
                "test.event",
                record_id="MRN-00012345",  # phi-fixture-ok: adversarial, proves hashing
                patient_id="PT-999",
            )
        )
        assert out["record_id"] != "MRN-00012345"  # phi-fixture-ok: adversarial
        assert out["patient_id"] != "PT-999"
        assert len(out["record_id"]) == 16

    def test_hashing_is_deterministic_for_log_correlation(self) -> None:
        log = get_logger("test.module")
        first = _capture(lambda: log.info("e", record_id="X"))
        second = _capture(lambda: log.info("e", record_id="X"))
        assert first["record_id"] == second["record_id"]

    def test_different_ids_hash_differently(self) -> None:
        log = get_logger("test.module")
        a = _capture(lambda: log.info("e", record_id="A"))
        b = _capture(lambda: log.info("e", record_id="B"))
        assert a["record_id"] != b["record_id"]

    def test_long_unlisted_string_is_truncated(self) -> None:
        log = get_logger("test.module")
        out = _capture(lambda: log.info("e", unexpected=("A" * 500)))
        assert len(out["unexpected"]) < 500
        assert out["unexpected"].endswith("...[truncated]")

    def test_short_string_is_not_truncated(self) -> None:
        log = get_logger("test.module")
        out = _capture(lambda: log.info("e", short="hello"))
        assert out["short"] == "hello"

    def test_logger_name_is_bound(self) -> None:
        """Regression guard: get_logger(name) originally never used `name`
        at all -- caught by ruff's ARG001, not by any test at the time."""
        log = get_logger("my.specific.module")
        out = _capture(lambda: log.info("e"))
        assert out["logger_name"] == "my.specific.module"


class TestAllowPhiToggle:
    def test_allow_phi_true_disables_redaction(self) -> None:
        configure_logging(allow_phi=True)
        log = get_logger("test.module")
        out = _capture(lambda: log.info("e", raw_text="NOW VISIBLE"))
        assert out["raw_text"] == "NOW VISIBLE"

    def test_allow_phi_true_logs_a_warning_immediately(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            configure_logging(allow_phi=True)
        lines = [json.loads(x) for x in buf.getvalue().strip().splitlines() if x]
        assert any(
            entry["event"] == "logging.phi_redaction_disabled" for entry in lines
        )

    def test_order_independence_pre_existing_logger_respects_later_toggle(self) -> None:
        """The property that would break a naive "bake the decision into
        which processors run" implementation: a logger created BEFORE
        allow_phi=True is set must still stop redacting once it's set."""
        log = get_logger("test.module")  # created while safe default is active
        configure_logging(allow_phi=True)
        out = _capture(lambda: log.info("e", raw_text="VISIBLE NOW"))
        assert out["raw_text"] == "VISIBLE NOW"

    def test_toggling_back_to_false_restores_redaction(self) -> None:
        log = get_logger("test.module")
        configure_logging(allow_phi=True)
        configure_logging(allow_phi=False)
        out = _capture(lambda: log.info("e", raw_text="SHOULD BE GONE"))
        assert "raw_text" not in out
