"""Shared contract every registered BasePreprocessor implementation must satisfy.

``_make_record`` is the same per-key fixture extension point already
established for the loader and chunker contract suites: ``DeidPreprocessor``
(``preprocessor.general.deidentify``) genuinely reads ``record.patient_ref``,
which the shared, generic ``FixtureRecord`` has no reason to carry.
"""

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


def _make_record(key: str, record_id: str, text: str) -> Any:
    """A valid, minimal record for `key`, matching whatever RecordT that
    preprocessor actually declares. Falls back to the shared generic
    FixtureRecord for any key not listed here."""
    if key == "preprocessor.general.deidentify" or key.startswith(
        "preprocessor.clinical_text."
    ):
        from openbtk.data.clinical_text.schemas import ClinicalTextRecord

        return ClinicalTextRecord(record_id=record_id, source="synthea", text=text)
    return FixtureRecord(record_id=record_id, text=text)


@pytest.mark.parametrize("key", PREPROCESSOR_REGISTRY.list_keys())
class TestPreprocessorContract:
    def test_process_returns_same_type(self, key: str) -> None:
        """process() returns a record of the same type it was given."""
        pre = _new_instance(key)
        record = _make_record(key, "r1", "hello")
        result = pre.process(record)
        assert type(result) is type(record)

    def test_process_is_stateless_across_calls(self, key: str) -> None:
        """The same input always produces the same output -- no hidden
        state accumulated between calls (docs/04_API_DESIGN.md section 3)."""
        pre = _new_instance(key)
        record = _make_record(key, "r1", "hello")
        first = pre.process(record)
        second = pre.process(record)
        assert first == second

    def test_process_stream_is_lazy_generator(self, key: str) -> None:
        """The default process_stream() maps process() over an iterable
        lazily -- a generator expression, not a materialised list."""
        pre = _new_instance(key)
        records = (_make_record(key, str(i), "x") for i in range(3))
        result = pre.process_stream(records)
        assert isinstance(result, types.GeneratorType) or hasattr(result, "__next__")

    def test_provenance_is_serialisable(self, key: str) -> None:
        pre = _new_instance(key)
        dumped = pre.provenance().model_dump_json()
        assert isinstance(dumped, str) and len(dumped) > 0
