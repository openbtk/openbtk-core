"""Unit tests for openbtk.core.plugins."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import openbtk.core.plugins as plugins_module
from openbtk.core.plugins import load_plugins


@pytest.fixture(autouse=True)
def _reset_loaded_flag() -> None:
    """Reset module-level _loaded flag before/after each test for isolation."""
    plugins_module._loaded = False
    yield
    plugins_module._loaded = False


def test_load_plugins_is_idempotent() -> None:
    with patch("importlib.metadata.entry_points") as mock_eps:
        mock_eps.return_value = []
        first = load_plugins()
        second = load_plugins()

    assert first == []
    assert second == []  # second call is a no-op due to _loaded flag
    mock_eps.assert_called_once()


def test_load_plugins_calls_registered_callable() -> None:
    mock_fn = MagicMock()
    mock_entry_point = MagicMock()
    mock_entry_point.name = "my_plugin"
    mock_entry_point.value = "my_package.module:register"
    mock_entry_point.load.return_value = mock_fn

    with patch("importlib.metadata.entry_points") as mock_eps:
        mock_eps.return_value = [mock_entry_point]
        loaded = load_plugins()

    assert loaded == ["my_plugin"]
    mock_fn.assert_called_once()


def test_load_plugins_handles_failed_plugin_gracefully() -> None:
    mock_entry_point = MagicMock()
    mock_entry_point.name = "broken_plugin"
    mock_entry_point.value = "broken.module:register"
    mock_entry_point.load.side_effect = ImportError("module not found")

    with patch("importlib.metadata.entry_points") as mock_eps:
        mock_eps.return_value = [mock_entry_point]
        loaded = load_plugins()

    # Failed plugin should not crash discovery, just be excluded
    assert loaded == []


def test_load_plugins_discovery_failure_returns_empty() -> None:
    with patch("importlib.metadata.entry_points") as mock_eps:
        mock_eps.side_effect = RuntimeError("entry point system unavailable")
        loaded = load_plugins()

    assert loaded == []


def test_load_plugins_mixed_success_and_failure() -> None:
    good_fn = MagicMock()
    good_ep = MagicMock()
    good_ep.name = "good_plugin"
    good_ep.value = "good.module:register"
    good_ep.load.return_value = good_fn

    bad_ep = MagicMock()
    bad_ep.name = "bad_plugin"
    bad_ep.value = "bad.module:register"
    bad_ep.load.side_effect = ImportError("boom")

    with patch("importlib.metadata.entry_points") as mock_eps:
        mock_eps.return_value = [good_ep, bad_ep]
        loaded = load_plugins()

    assert loaded == ["good_plugin"]
    good_fn.assert_called_once()
