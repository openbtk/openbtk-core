"""Unit tests for openbtk.core._lazy.require().

Not covered by the contract suite at all (0% coverage baseline) -- no
reference implementation needs an optional dependency, since they are
deliberately stdlib-only. This is the only place that behaviour is tested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from openbtk.core._lazy import _CACHE, require
from openbtk.core.errors import MissingDependencyError

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _clear_cache() -> Iterator[None]:
    """The module-level cache must not leak state between tests."""
    _CACHE.clear()
    yield
    _CACHE.clear()


def test_require_returns_the_real_module_for_stdlib() -> None:
    mod = require("json", extra="never-missing")
    assert mod.__name__ == "json"


def test_require_caches_across_calls() -> None:
    first = require("json", extra="never-missing")
    second = require("json", extra="never-missing")
    assert first is second


def test_require_raises_missing_dependency_error_for_nonexistent_module() -> None:
    with pytest.raises(MissingDependencyError) as exc_info:
        require("this_module_definitely_does_not_exist_xyz", extra="fake-extra")
    assert "fake-extra" in str(exc_info.value)
    assert "pip install" in str(exc_info.value)


def test_missing_dependency_error_context_names_module_and_extra() -> None:
    with pytest.raises(MissingDependencyError) as exc_info:
        require("this_module_definitely_does_not_exist_xyz", extra="fake-extra")
    assert (
        exc_info.value.context["module"] == "this_module_definitely_does_not_exist_xyz"
    )
    assert exc_info.value.context["extra"] == "fake-extra"


def test_require_chains_the_original_import_error() -> None:
    with pytest.raises(MissingDependencyError) as exc_info:
        require("this_module_definitely_does_not_exist_xyz", extra="fake-extra")
    assert isinstance(exc_info.value.__cause__, ImportError)


def test_failed_lookup_is_not_cached() -> None:
    """A missing module must not poison the cache -- a later successful
    import of something with the same (nonexistent) name should still be
    attempted fresh, not short-circuited by a cached failure."""
    with pytest.raises(MissingDependencyError):
        require("this_module_definitely_does_not_exist_xyz", extra="fake-extra")
    assert "this_module_definitely_does_not_exist_xyz" not in _CACHE
