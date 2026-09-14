"""Unit tests for openbtk.core.base beyond what the contract suite covers.

The contract suite exercises every concrete base-class default through real
registered implementations; this file covers the one path that cannot be
reached that way -- _openbtk_version()'s PackageNotFoundError fallback,
which only fires in an environment where openbtk's own distribution
metadata is unavailable (never true during a real test run, since the
package is always installed via `pip install -e .` first).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from typing import TYPE_CHECKING

from openbtk.core import base

if TYPE_CHECKING:
    import pytest


def test_openbtk_version_falls_back_when_package_metadata_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(base, "version", _raise)
    assert base._openbtk_version() == "0.0.0.dev0+unknown"
