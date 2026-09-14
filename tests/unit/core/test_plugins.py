"""Unit tests for openbtk.core.plugins.load_plugins().

Originally verified end-to-end against a real, separately pip-installed
throwaway package with genuine entry points in the "openbtk.providers"
group (one registering a real loader, one deliberately raising) --
converted to permanent tests here using monkeypatched entry points instead,
so this suite does not depend on external package installation/removal at
test time. The plugin-name/deny-list bug and the +SKIP semantics bug found
during that manual verification are exactly why plugin_name= (not name=)
and the redirect_stdout-based doctest exist in the shipped module; this
file guards the discovery/failure/idempotency logic those fixes sit on top of.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from importlib.metadata import EntryPoint
from typing import TYPE_CHECKING, Any

import pytest

import openbtk.core.plugins as plugins_module
from openbtk.core.plugins import load_plugins

if TYPE_CHECKING:
    from collections.abc import Iterator


def _captured_log_lines(buf: io.StringIO) -> list[dict[str, Any]]:
    """Parse the JSON log lines written to `buf` during a captured call."""
    return [json.loads(line) for line in buf.getvalue().strip().splitlines() if line]


@pytest.fixture(autouse=True)
def _reset_loaded_flag() -> Iterator[None]:
    """load_plugins() runs at most once per process by design -- each test
    needs a fresh, unfired flag, and must leave it fired-or-not exactly as
    it found it so later tests (in this file or elsewhere) aren't affected."""
    original = plugins_module._loaded
    plugins_module._loaded = False
    yield
    plugins_module._loaded = original


def _fake_entry_points(*eps: EntryPoint) -> Any:
    """A stand-in for importlib.metadata.entry_points(group=...) returning
    exactly the given entry points, regardless of the group argument."""

    def _fn(*, group: str) -> tuple[EntryPoint, ...]:
        return eps

    return _fn


def test_runs_at_most_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def _tracking_entry_points(*, group: str) -> tuple[EntryPoint, ...]:
        calls.append(group)
        return ()

    monkeypatch.setattr(plugins_module, "entry_points", _tracking_entry_points)
    load_plugins()
    load_plugins()
    load_plugins()
    assert len(calls) == 1


def test_discovers_and_calls_a_good_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """.load() is mocked to return a callable directly rather than
    resolving a real "module:function" value string -- this test guards
    OUR code's "call whatever .load() returns" logic, not importlib's own
    entry-point resolution mechanism, which is stdlib. The genuine
    end-to-end path (a real installed package, a real resolvable entry
    point) was verified manually during development; see this module's
    docstring."""
    registered = []
    good_ep = EntryPoint(name="good_plugin", value="x:y", group="openbtk.providers")
    monkeypatch.setattr(plugins_module, "entry_points", _fake_entry_points(good_ep))
    monkeypatch.setattr(
        good_ep.__class__, "load", lambda self: lambda: registered.append("called")
    )
    load_plugins()
    assert registered == ["called"]


def test_a_broken_plugin_logs_a_warning_and_does_not_crash_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise() -> None:
        raise RuntimeError("deliberately broken")

    broken_ep = EntryPoint(name="broken_plugin", value="x:y", group="openbtk.providers")
    monkeypatch.setattr(plugins_module, "entry_points", _fake_entry_points(broken_ep))
    monkeypatch.setattr(broken_ep.__class__, "load", lambda self: _raise)

    buf = io.StringIO()
    with redirect_stdout(buf):
        load_plugins()  # must not raise
    log_lines = _captured_log_lines(buf)
    assert any(entry.get("event") == "plugins.load_failed" for entry in log_lines)


def test_broken_plugin_warning_names_the_plugin_not_dropped_by_redaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard for the real bug found during manual verification:
    logging the failing plugin as name=ep.name is silently dropped by the
    PHI deny-list ("name" is in it). Must use plugin_name= instead."""

    def _raise() -> None:
        raise RuntimeError("deliberately broken")

    broken_ep = EntryPoint(
        name="identifiable_broken_plugin", value="x:y", group="openbtk.providers"
    )
    monkeypatch.setattr(plugins_module, "entry_points", _fake_entry_points(broken_ep))
    monkeypatch.setattr(broken_ep.__class__, "load", lambda self: _raise)

    buf = io.StringIO()
    with redirect_stdout(buf):
        load_plugins()
    log_lines = _captured_log_lines(buf)
    failure = next(
        entry for entry in log_lines if entry.get("event") == "plugins.load_failed"
    )
    assert failure.get("plugin_name") == "identifiable_broken_plugin"
    assert "name" not in failure  # confirms the deny-listed key truly never appears


def test_one_broken_plugin_does_not_prevent_others_from_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered = []

    def _raise() -> None:
        raise RuntimeError("broken")

    broken_ep = EntryPoint(name="broken", value="x:y", group="openbtk.providers")
    good_ep = EntryPoint(name="good", value="x:y", group="openbtk.providers")
    monkeypatch.setattr(
        plugins_module, "entry_points", _fake_entry_points(broken_ep, good_ep)
    )

    def _fake_load(self: EntryPoint) -> Any:
        return _raise if self.name == "broken" else (lambda: registered.append("ok"))

    monkeypatch.setattr(EntryPoint, "load", _fake_load)
    load_plugins()
    assert registered == ["ok"]


def test_discovery_itself_failing_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even importlib.metadata.entry_points() itself raising must not
    propagate -- "must never prevent import openbtk from succeeding" is
    unconditional (docs/04_API_DESIGN.md section 10)."""

    def _raise(*, group: str) -> Any:
        raise RuntimeError("corrupted environment")

    monkeypatch.setattr(plugins_module, "entry_points", _raise)
    load_plugins()  # must not raise
