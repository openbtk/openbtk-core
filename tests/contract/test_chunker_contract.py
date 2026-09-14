"""Shared contract every registered BaseChunker implementation must satisfy.

The single most architecturally important guarantee in this codebase
(ADR-0004): ``chunk()`` returns an ``Iterator``, never a ``list``. A single
large record can produce thousands of chunks; returning a list satisfies
mypy and silently defeats the whole streaming memory guarantee the project
is built around.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from openbtk.core.registry import CHUNKER_REGISTRY

from .conftest import FixtureChunk, FixtureRecord

if TYPE_CHECKING:
    from openbtk.core.base import BaseChunker


def _new_instance(key: str) -> BaseChunker[Any, Any]:
    return CHUNKER_REGISTRY.create(key)


@pytest.mark.parametrize("key", CHUNKER_REGISTRY.list_keys())
class TestChunkerContract:
    def test_chunk_returns_iterator(self, key: str) -> None:
        """chunk() returns a genuine Iterator, not a list (ADR-0004)."""
        chunker = _new_instance(key)
        record = FixtureRecord(record_id="r1", text="one two three")
        result = chunker.chunk(record)
        assert isinstance(result, Iterator), (
            f"{key}: chunk() returned {type(result).__name__}, not an "
            "Iterator. This is the single most important contract in the "
            "project (ADR-0004) -- a list return type-checks fine and "
            "silently breaks the streaming memory guarantee."
        )

    def test_empty_input_yields_nothing_without_raising(self, key: str) -> None:
        """A record below the minimum chunkable length yields zero chunks --
        that is valid, not an error (docs/04_API_DESIGN.md section 3)."""
        chunker = _new_instance(key)
        record = FixtureRecord(record_id="r1", text="")
        chunks = list(chunker.chunk(record))
        assert chunks == []

    def test_chunks_validate_against_schema(self, key: str) -> None:
        chunker = _new_instance(key)
        record = FixtureRecord(record_id="r1", text="alpha beta")
        chunks = list(chunker.chunk(record))
        assert chunks, f"{key}: yielded nothing for a non-empty record"
        assert all(isinstance(c, BaseModel) for c in chunks)

    def test_provenance_is_serialisable(self, key: str) -> None:
        chunker = _new_instance(key)
        dumped = chunker.provenance().model_dump_json()
        assert isinstance(dumped, str) and len(dumped) > 0


# ---------------------------------------------------------------------------
# Meta-test: proves the Iterator-not-list check actually catches a violation.
# Never registered globally.
# ---------------------------------------------------------------------------


class _ListReturningBrokenChunker:
    """Deliberately violates ADR-0004 by returning a list. This is exactly
    the mistake that satisfies a type checker but breaks streaming."""

    def chunk(self, record: FixtureRecord) -> list[FixtureChunk]:
        return [
            FixtureChunk(
                chunk_id=f"{record.record_id}-{i}", record_id=record.record_id, text=w
            )
            for i, w in enumerate(record.text.split())
        ]


def test_iterator_check_catches_a_list_returning_violation() -> None:
    broken = _ListReturningBrokenChunker()
    record = FixtureRecord(record_id="r1", text="a b c")
    result = broken.chunk(record)
    assert not isinstance(result, Iterator), (
        "sanity check: a plain list must NOT be an Iterator, or this "
        "meta-test proves nothing"
    )
