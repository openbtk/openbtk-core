"""Shared contract every registered BaseSegmenter implementation must satisfy.

Semantically identical to BaseChunker's contract (see
test_chunker_contract.py) -- segment() is chunk()'s non-text-modality twin,
used for imaging patches, signal windows, video clips. Same ADR-0004
Iterator guarantee applies.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from openbtk.core.registry import SEGMENTER_REGISTRY

from .conftest import FixtureRecord

if TYPE_CHECKING:
    from openbtk.core.base import BaseSegmenter


def _new_instance(key: str) -> BaseSegmenter[Any, Any]:
    return SEGMENTER_REGISTRY.create(key)


@pytest.mark.parametrize("key", SEGMENTER_REGISTRY.list_keys())
class TestSegmenterContract:
    def test_segment_returns_iterator(self, key: str) -> None:
        """segment() returns a genuine Iterator, not a list (ADR-0004)."""
        segmenter = _new_instance(key)
        record = FixtureRecord(record_id="r1", text="one two three")
        result = segmenter.segment(record)
        assert isinstance(result, Iterator), (
            f"{key}: segment() returned {type(result).__name__}, not an "
            "Iterator -- same ADR-0004 violation as a chunker returning a list."
        )

    def test_empty_input_yields_nothing_without_raising(self, key: str) -> None:
        segmenter = _new_instance(key)
        record = FixtureRecord(record_id="r1", text="")
        segments = list(segmenter.segment(record))
        assert segments == []

    def test_segments_validate_against_schema(self, key: str) -> None:
        segmenter = _new_instance(key)
        record = FixtureRecord(record_id="r1", text="alpha beta")
        segments = list(segmenter.segment(record))
        assert segments, f"{key}: yielded nothing for a non-empty record"
        assert all(isinstance(s, BaseModel) for s in segments)

    def test_provenance_is_serialisable(self, key: str) -> None:
        segmenter = _new_instance(key)
        dumped = segmenter.provenance().model_dump_json()
        assert isinstance(dumped, str) and len(dumped) > 0
