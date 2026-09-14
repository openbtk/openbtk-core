"""Shared contract every registered BasePreprocessor implementation must satisfy."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

import pytest

from openbtk.core.registry import PREPROCESSOR_REGISTRY

from .conftest import FixtureRecord

if TYPE_CHECKING:
    from openbtk.core.base import BasePreprocessor


def _new_instance(key: str) -> BasePreprocessor[Any]:
    return PREPROCESSOR_REGISTRY.create(key)


@pytest.mark.parametrize("key", PREPROCESSOR_REGISTRY.list_keys())
class TestPreprocessorContract:
    def test_process_returns_same_type(self, key: str) -> None:
        """process() returns a record of the same type it was given."""
        pre = _new_instance(key)
        record = FixtureRecord(record_id="r1", text="hello")
        result = pre.process(record)
        assert type(result) is type(record)

    def test_process_is_stateless_across_calls(self, key: str) -> None:
        """The same input always produces the same output -- no hidden
        state accumulated between calls (docs/04_API_DESIGN.md section 3)."""
        pre = _new_instance(key)
        record = FixtureRecord(record_id="r1", text="hello")
        first = pre.process(record)
        second = pre.process(record)
        assert first == second

    def test_process_stream_is_lazy_generator(self, key: str) -> None:
        """The default process_stream() maps process() over an iterable
        lazily -- a generator expression, not a materialised list."""
        pre = _new_instance(key)
        records = (FixtureRecord(record_id=str(i), text="x") for i in range(3))
        result = pre.process_stream(records)
        assert isinstance(result, types.GeneratorType) or hasattr(result, "__next__")

    def test_provenance_is_serialisable(self, key: str) -> None:
        pre = _new_instance(key)
        dumped = pre.provenance().model_dump_json()
        assert isinstance(dumped, str) and len(dumped) > 0
