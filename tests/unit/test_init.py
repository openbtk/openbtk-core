"""Unit test for openbtk/__init__.py's own version-detection fallback.

Same pattern, same reason, as tests/unit/core/test_base.py's coverage of
_openbtk_version()'s PackageNotFoundError fallback: only reachable in an
environment where openbtk's own distribution metadata is unavailable, never
true during a real test run since the package is always installed via
`pip install -e .` first. Found as a genuine, pre-existing coverage gap
(82%, lines 75-76) while verifying tests/security/ -- top-level __init__.py
has the identical fallback as core/base.py's _openbtk_version but no test
of its own.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from typing import TYPE_CHECKING

import openbtk

if TYPE_CHECKING:
    import pytest


def test_detect_version_falls_back_when_package_metadata_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(openbtk, "version", _raise)
    assert openbtk._detect_version() == "0.0.0.dev0+unknown"
