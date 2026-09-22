"""Checkpoint and resume for long pipeline runs (FR-L-05).

A run that loads ten million notes and dies at note nine million should not start over.
Passing ``checkpoint_path=`` to :meth:`~openbtk.pipelines.pipeline.Pipeline.run` makes
the executor periodically write how far each root loader has read; running the same
pipeline again with the same path picks up from there instead of the beginning.

**What is checkpointed, precisely: loader position, not pipeline progress.** Only root
loader steps are tracked -- "record 4,512,003 of this source has been read" -- not
whether every downstream step finished with it. Resuming re-loads from that position and
runs it through the *whole* pipeline again, redoing any downstream work that had already
happened. This is safe because every step here is a pure function of its input (the same
record in gives the same output), never because the executor remembers partial
downstream state -- it does not.

**The guarantee is at-least-once, not exactly-once.** A checkpoint is written every
``checkpoint_interval`` records (default 1000), so a crash between saves can reprocess
up to that many records again; it never skips one that was not genuinely already read. A
completed run (``RunManifest.status == "success"``) deletes its checkpoint file, since
nothing is left to resume.

**Resuming reads and discards the skipped records; it does not seek.** The loader's
``load()`` is called again from the start of the same source and the first *N* items it
produces are dropped before the rest reach the pipeline. This is correct only when the
source yields the same items in the same order across runs (a stable file, not a query
against data that changes between attempts), and it is not free: whatever work the
loader does per record before discarding it (parsing a CSV row, say) still happens for
every skipped record.

Only the loader's own step, in :class:`~openbtk.core.provenance.StepProvenance`, records
how many of its inputs were skipped this way, in ``resumed_from``.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from openbtk.core.errors import ConfigError


class Checkpoint(BaseModel):
    """Resume state for one pipeline: how far each root loader step has read.

    Example:
        >>> from datetime import UTC, datetime
        >>> Checkpoint(
        ...     pipeline_name="ingest", run_id="a1b2c3",
        ...     loader_counts={"load": 4512003},
        ...     saved_at=datetime(2026, 1, 1, tzinfo=UTC),
        ... ).loader_counts["load"]
        4512003
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    checkpoint_version: str = Field(
        "1.0.0",
        description="Schema version of this file, independent of openbtk's own.",
    )
    pipeline_name: str = Field(
        ..., min_length=1, description="Must match the config being resumed."
    )
    run_id: str = Field(..., min_length=1, description="The run that last saved this.")
    loader_counts: dict[str, int] = Field(
        ..., description="Loader step id -> records of its source already read."
    )
    saved_at: datetime


def save_checkpoint(path: str | Path, checkpoint: Checkpoint) -> None:
    """Write ``checkpoint`` to ``path`` atomically (write to a temp file, then rename),
    so a crash mid-write never leaves a truncated, unreadable checkpoint file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    tmp.write_text(checkpoint.model_dump_json(indent=2) + "\n", encoding="utf-8")
    tmp.replace(target)


def load_checkpoint(path: str | Path) -> Checkpoint:
    """Read a checkpoint file.

    Raises:
        ConfigError: If it cannot be read or parsed. Resuming from a checkpoint you
            cannot trust would risk silently skipping unprocessed records, so this
            refuses rather than falling back to "start from the beginning."
    """
    target = Path(path)
    try:
        raw = target.read_text(encoding="utf-8")
        return Checkpoint.model_validate_json(raw)
    except (OSError, ValueError) as e:
        raise ConfigError(
            f"Could not read checkpoint file {target}: {e}",
            context={"checkpoint": str(target)},
        ) from e


class CheckpointState:
    """Tracks and periodically saves loader positions during one run.

    Not part of the public API: built by the executor from ``Pipeline.run``'s
    ``checkpoint_path``/``checkpoint_interval`` arguments.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        pipeline_name: str,
        run_id: str,
        interval: int,
        positions: dict[str, int],
    ) -> None:
        if interval < 1:
            raise ConfigError(
                "checkpoint_interval must be at least 1.",
                context={"checkpoint_interval": interval},
            )
        self._path = Path(path)
        self._pipeline_name = pipeline_name
        self._run_id = run_id
        self._interval = interval
        self._positions = dict(positions)

    @classmethod
    def load_or_start(
        cls,
        path: str | Path,
        *,
        pipeline_name: str,
        run_id: str,
        interval: int,
    ) -> CheckpointState:
        """Resume from ``path`` if it exists (checking it was saved for this same
        pipeline), or start fresh at position 0 for every loader."""
        target = Path(path)
        positions: dict[str, int] = {}
        if target.exists():
            saved = load_checkpoint(target)
            if saved.pipeline_name != pipeline_name:
                raise ConfigError(
                    f"Checkpoint {target} was saved for pipeline "
                    f"{saved.pipeline_name!r}, not {pipeline_name!r}. Use a "
                    "different --checkpoint path per pipeline, or delete the "
                    "file if this is intentional.",
                    context={"checkpoint": str(target), "pipeline": pipeline_name},
                )
            positions = dict(saved.loader_counts)
        return cls(
            target,
            pipeline_name=pipeline_name,
            run_id=run_id,
            interval=interval,
            positions=positions,
        )

    def start_position(self, step_id: str) -> int:
        """How many of ``step_id``'s source records were read before this run."""
        return self._positions.get(step_id, 0)

    def advance(self, step_id: str, position: int) -> None:
        """Record ``step_id``'s new position, saving every ``interval`` records.

        A save reflects every loader's last-known position, not only ``step_id``'s --
        the file always describes one consistent resume point for the whole pipeline.
        """
        self._positions[step_id] = position
        if position % self._interval == 0:
            self.save()

    def save(self) -> None:
        """Write the current positions now, regardless of the interval. Called once more
        after the run ends, so a failure between periodic saves does not lose progress
        beyond the last completed interval."""
        save_checkpoint(
            self._path,
            Checkpoint(
                pipeline_name=self._pipeline_name,
                run_id=self._run_id,
                loader_counts=dict(self._positions),
                saved_at=datetime.now(UTC),
            ),
        )

    def clear(self) -> None:
        """Remove the checkpoint file: a completed run leaves nothing to resume."""
        self._path.unlink(missing_ok=True)
