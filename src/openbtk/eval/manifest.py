"""``EvalManifest``: the provenance record an evaluation emits (FR-X-06).

An evaluation is a run like any other -- which model was scored, on which
data, when, and with what result -- and the same rule applies as for a
pipeline ``RunManifest`` (ADR-0005): it carries **identifiers and counts,
never content**. No question text, no answer text, no document text; a
manifest from a credentialed dataset must be as safe to file as one from a
synthetic corpus.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from openbtk.core.provenance import ComponentProvenance, DataDigest
from openbtk.core.schemas import JsonValue  # noqa: TC001 -- Pydantic field type

if TYPE_CHECKING:
    from collections.abc import Sequence

_HASH_CHUNK = 1 << 20


class EvalManifest(BaseModel):
    """The audit record of one evaluation run.

    Example:
        >>> m = EvalManifest(
        ...     eval_id="e1", kind="qa", started_at=datetime(2026, 1, 1, tzinfo=UTC),
        ...     ended_at=datetime(2026, 1, 1, tzinfo=UTC), report={"n": 3},
        ... )
        >>> m.manifest_version
        '1.0.0'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest_version: str = Field("1.0.0")
    eval_id: str = Field(..., min_length=1)
    kind: str = Field(
        ..., min_length=1, description='What was evaluated, e.g. "qa", "groundedness".'
    )
    started_at: datetime
    ended_at: datetime
    component: ComponentProvenance | None = Field(
        None, description="The component under test (e.g. the LLM provider), if any."
    )
    input_digests: list[DataDigest] = Field(default_factory=list)
    report: dict[str, JsonValue] = Field(
        ..., description="The evaluation's scores: counts and ratios only."
    )


def file_digest(path: str | Path, *, record_count: int) -> DataDigest:
    """A ``DataDigest`` for a local file, hashed in 1 MiB chunks (so an
    evaluation set is never held in memory just to be fingerprinted).

    Example:
        >>> import tempfile, pathlib
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = pathlib.Path(d) / "x.jsonl"
        ...     _ = p.write_text("{}")
        ...     digest = file_digest(p, record_count=1)
        >>> len(digest.sha256 or "")
        64
    """
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(_HASH_CHUNK):
            h.update(chunk)
    return DataDigest(uri=str(path), sha256=h.hexdigest(), record_count=record_count)


def build_manifest(
    kind: str,
    report: dict[str, JsonValue],
    *,
    started_at: datetime,
    ended_at: datetime | None = None,
    component: ComponentProvenance | None = None,
    input_digests: Sequence[DataDigest] = (),
) -> EvalManifest:
    """Assemble a manifest with a fresh id. ``ended_at`` defaults to now."""
    return EvalManifest(
        eval_id=uuid.uuid4().hex,
        kind=kind,
        started_at=started_at,
        ended_at=ended_at or datetime.now(UTC),
        component=component,
        input_digests=list(input_digests),
        report=report,
    )
