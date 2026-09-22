"""Checkpoint and resume (FR-L-05): the ``Checkpoint``/``CheckpointState`` machinery on
its own, then a real crash-and-resume through ``Pipeline.run``."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar

import pytest

from openbtk.core.base import BaseLoader, BasePreprocessor
from openbtk.core.config import PipelineConfig, StepConfig
from openbtk.core.errors import ConfigError, LoaderError
from openbtk.core.registry import LOADER_REGISTRY, PREPROCESSOR_REGISTRY
from openbtk.pipelines import Pipeline
from openbtk.pipelines.checkpoint import (
    Checkpoint,
    CheckpointState,
    load_checkpoint,
    save_checkpoint,
)

from .test_executor import _Rec

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_T = datetime(2026, 1, 1, tzinfo=UTC)


def _checkpoint(**kwargs: object) -> Checkpoint:
    defaults: dict[str, object] = {
        "pipeline_name": "p",
        "run_id": "r1",
        "loader_counts": {"load": 5},
        "saved_at": _T,
    }
    return Checkpoint(**{**defaults, **kwargs})  # type: ignore[arg-type]


class TestCheckpointRoundTrip:
    def test_save_then_load(self, tmp_path: Path) -> None:
        path = tmp_path / "ck.json"
        save_checkpoint(path, _checkpoint(loader_counts={"load": 42}))
        assert load_checkpoint(path).loader_counts == {"load": 42}

    def test_a_save_overwrites_atomically_and_leaves_no_temp_file(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "ck.json"
        save_checkpoint(path, _checkpoint(loader_counts={"load": 1}))
        save_checkpoint(path, _checkpoint(loader_counts={"load": 2}))
        assert load_checkpoint(path).loader_counts == {"load": 2}
        assert list(tmp_path.iterdir()) == [path]

    def test_save_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "a" / "b" / "ck.json"
        save_checkpoint(path, _checkpoint())
        assert path.exists()

    def test_loading_a_missing_file_is_a_config_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="Could not read"):
            load_checkpoint(tmp_path / "nope.json")

    def test_loading_garbage_is_a_config_error_not_a_silent_restart(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "ck.json"
        path.write_text("not json", encoding="utf-8")
        with pytest.raises(ConfigError, match="Could not read"):
            load_checkpoint(path)

    def test_the_file_is_plain_readable_json(self, tmp_path: Path) -> None:
        path = tmp_path / "ck.json"
        save_checkpoint(path, _checkpoint(loader_counts={"load": 9}))
        assert '"load": 9' in path.read_text(encoding="utf-8")


class TestCheckpointState:
    def test_a_missing_file_starts_at_zero(self, tmp_path: Path) -> None:
        state = CheckpointState.load_or_start(
            tmp_path / "ck.json", pipeline_name="p", run_id="r", interval=10
        )
        assert state.start_position("load") == 0

    def test_an_existing_file_is_resumed(self, tmp_path: Path) -> None:
        path = tmp_path / "ck.json"
        save_checkpoint(path, _checkpoint(loader_counts={"load": 30}))
        state = CheckpointState.load_or_start(
            path, pipeline_name="p", run_id="r2", interval=10
        )
        assert state.start_position("load") == 30

    def test_a_pipeline_name_mismatch_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "ck.json"
        save_checkpoint(path, _checkpoint(pipeline_name="other"))
        with pytest.raises(ConfigError, match="other"):
            CheckpointState.load_or_start(
                path, pipeline_name="p", run_id="r", interval=10
            )

    def test_an_unknown_loader_id_defaults_to_zero(self, tmp_path: Path) -> None:
        path = tmp_path / "ck.json"
        save_checkpoint(path, _checkpoint(loader_counts={"other_step": 5}))
        state = CheckpointState.load_or_start(
            path, pipeline_name="p", run_id="r", interval=10
        )
        assert state.start_position("load") == 0

    def test_advance_saves_only_on_the_interval(self, tmp_path: Path) -> None:
        path = tmp_path / "ck.json"
        state = CheckpointState(
            path, pipeline_name="p", run_id="r", interval=5, positions={}
        )
        for position in range(1, 5):
            state.advance("load", position)
            assert not path.exists()
        state.advance("load", 5)
        assert load_checkpoint(path).loader_counts == {"load": 5}

    def test_save_writes_regardless_of_the_interval(self, tmp_path: Path) -> None:
        path = tmp_path / "ck.json"
        state = CheckpointState(
            path, pipeline_name="p", run_id="r", interval=1000, positions={}
        )
        state.advance("load", 3)
        assert not path.exists()
        state.save()
        assert load_checkpoint(path).loader_counts == {"load": 3}

    def test_clear_removes_the_file_and_is_safe_if_already_gone(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "ck.json"
        state = CheckpointState(
            path, pipeline_name="p", run_id="r", interval=1, positions={}
        )
        state.advance("load", 1)
        assert path.exists()
        state.clear()
        assert not path.exists()
        state.clear()  # a second call does not raise

    def test_a_save_reflects_every_loaders_last_known_position(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "ck.json"
        state = CheckpointState(
            path, pipeline_name="p", run_id="r", interval=2, positions={}
        )
        state.advance("a", 1)
        state.advance("b", 2)  # triggers a save; "a" is included at its last value
        assert load_checkpoint(path).loader_counts == {"a": 1, "b": 2}

    def test_a_bad_interval_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="checkpoint_interval"):
            CheckpointState(
                tmp_path / "ck.json",
                pipeline_name="p",
                run_id="r",
                interval=0,
                positions={},
            )


# ---------------------------------------------------------------- through Pipeline.run


@LOADER_REGISTRY.register("loader.general.checkpoint_test_source")
class _CrashableLoader(BaseLoader[int, _Rec]):
    """Yields ``source`` records, raising just before a configured index -- so items
    before it are genuinely emitted and the crash is reproducible."""

    crash_at: ClassVar[int | None] = None

    def load(self, source: int) -> Iterator[_Rec]:
        for i in range(source):
            if self.crash_at is not None and i == self.crash_at:
                raise LoaderError("simulated crash", context={"at": i})
            yield _Rec(record_id=str(i), text=f"r{i}")


@pytest.fixture(autouse=True)
def _no_crash() -> Iterator[None]:
    _CrashableLoader.crash_at = None
    yield
    _CrashableLoader.crash_at = None


def _config(count: int, checkpoint_step_id: str = "load") -> PipelineConfig:
    return PipelineConfig(
        name="checkpoint-demo",
        steps=[
            StepConfig(
                id=checkpoint_step_id,
                type="loader.general.checkpoint_test_source",
                params={"source": count},
            )
        ],
    )


class TestResumeThroughAPipeline:
    def test_a_successful_run_deletes_the_checkpoint(self, tmp_path: Path) -> None:
        path = tmp_path / "ck.json"
        manifest = Pipeline.from_config(_config(25)).run(
            checkpoint_path=path, checkpoint_interval=5
        )
        assert manifest.status == "success"
        assert manifest.steps[0].records_out == 25
        assert manifest.steps[0].resumed_from == 0
        assert not path.exists()

    def test_a_crash_leaves_the_exact_position_not_rounded_to_the_interval(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "ck.json"
        _CrashableLoader.crash_at = 17
        manifest = Pipeline.from_config(_config(30)).run(
            checkpoint_path=path, checkpoint_interval=5
        )
        assert manifest.status == "failed"
        assert manifest.steps[0].records_out == 17
        assert load_checkpoint(path).loader_counts == {"load": 17}

    def test_resuming_picks_up_exactly_where_it_left_off(self, tmp_path: Path) -> None:
        path = tmp_path / "ck.json"
        _CrashableLoader.crash_at = 17
        Pipeline.from_config(_config(30)).run(
            checkpoint_path=path, checkpoint_interval=5
        )
        _CrashableLoader.crash_at = None  # the "bug" is "fixed" before the retry
        manifest = Pipeline.from_config(_config(30)).run(
            checkpoint_path=path, checkpoint_interval=5
        )
        assert manifest.status == "success"
        assert manifest.steps[0].records_out == 13  # 30 - 17
        assert manifest.steps[0].resumed_from == 17
        assert not path.exists()

    def test_resumed_records_are_never_reprocessed(self, tmp_path: Path) -> None:
        path = tmp_path / "ck.json"
        seen: list[str] = []

        @PREPROCESSOR_REGISTRY.register("preprocessor.general.checkpoint_test_record")
        class _Recorder(BasePreprocessor[_Rec]):
            def process(self, record: _Rec) -> _Rec:
                seen.append(record.text)
                return record

        config = PipelineConfig(
            name="checkpoint-demo",
            steps=[
                StepConfig(
                    id="load",
                    type="loader.general.checkpoint_test_source",
                    params={"source": 20},
                ),
                StepConfig(
                    id="record",
                    type="preprocessor.general.checkpoint_test_record",
                    after=["load"],
                ),
            ],
        )
        _CrashableLoader.crash_at = 8
        Pipeline.from_config(config).run(checkpoint_path=path, checkpoint_interval=3)
        assert seen == [f"r{i}" for i in range(8)]
        seen.clear()
        _CrashableLoader.crash_at = None
        Pipeline.from_config(config).run(checkpoint_path=path, checkpoint_interval=3)
        assert seen == [f"r{i}" for i in range(8, 20)]

    def test_an_unrelated_config_change_after_a_crash_still_resumes_by_position(
        self, tmp_path: Path
    ) -> None:
        """A real, disclosed limitation: resume is by count, not content. Changing the
        source between attempts silently resumes from the wrong records; this test
        pins that this is what happens, not to endorse it."""
        path = tmp_path / "ck.json"
        _CrashableLoader.crash_at = 5
        Pipeline.from_config(_config(10)).run(
            checkpoint_path=path, checkpoint_interval=2
        )
        _CrashableLoader.crash_at = None
        # A different count now (as if the source changed): resume still starts at 5.
        manifest = Pipeline.from_config(_config(9)).run(
            checkpoint_path=path, checkpoint_interval=2
        )
        assert manifest.status == "success"
        assert manifest.steps[0].records_out == 4  # 9 - 5

    def test_no_checkpoint_path_means_no_file_and_no_resume_bookkeeping(
        self, tmp_path: Path
    ) -> None:
        manifest = Pipeline.from_config(_config(5)).run()
        assert manifest.steps[0].resumed_from == 0
        assert list(tmp_path.iterdir()) == []

    def test_a_pipeline_name_mismatch_fails_the_run_not_the_caller(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "ck.json"
        save_checkpoint(
            path,
            Checkpoint(
                pipeline_name="a-different-pipeline",
                run_id="r",
                loader_counts={"load": 3},
                saved_at=_T,
            ),
        )
        manifest = Pipeline.from_config(_config(5)).run(checkpoint_path=path)
        assert manifest.status == "failed"
        assert "a-different-pipeline" in (manifest.error or "")

    def test_a_fan_out_pipeline_checkpoints_only_the_loader(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "ck.json"
        config = PipelineConfig(
            name="checkpoint-demo",
            steps=[
                StepConfig(
                    id="load",
                    type="loader.general.checkpoint_test_source",
                    params={"source": 12},
                ),
                StepConfig(
                    id="a",
                    type="preprocessor.general.pipeline_test_upper",
                    after=["load"],
                ),
                StepConfig(
                    id="b",
                    type="preprocessor.general.pipeline_test_upper",
                    after=["load"],
                ),
            ],
        )
        manifest = Pipeline.from_config(config).run(
            checkpoint_path=path, checkpoint_interval=4
        )
        assert manifest.status == "success"
        by_id = {s.step_id: s for s in manifest.steps}
        assert by_id["load"].resumed_from == 0
        assert by_id["a"].resumed_from == 0 and by_id["b"].resumed_from == 0
        assert by_id["a"].records_out == 12 and by_id["b"].records_out == 12

    def test_resumed_from_is_recorded_on_the_manifest(self, tmp_path: Path) -> None:
        path = tmp_path / "ck.json"
        save_checkpoint(
            path,
            _checkpoint(pipeline_name="checkpoint-demo", loader_counts={"load": 4}),
        )
        manifest = Pipeline.from_config(_config(10)).run(checkpoint_path=path)
        assert manifest.steps[0].resumed_from == 4
        assert '"resumed_from":4' in manifest.model_dump_json()
