"""Fan-out and fan-in in the streaming executor (FR-L-04).

A step may feed several steps (each gets every record) and may read several steps (their
streams are interleaved). The tests assert what the manifest says, what each branch
actually saw, and that memory stays bounded: the branches advance in lock-step, so a
broadcast never buffers the corpus."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from openbtk.core.base import BaseGuardrail, BaseLoader, BasePreprocessor
from openbtk.core.config import PipelineConfig, StepConfig
from openbtk.core.registry import (
    GUARDRAIL_REGISTRY,
    LOADER_REGISTRY,
    PREPROCESSOR_REGISTRY,
)
from openbtk.core.schemas import GuardrailResult, GuardrailSeverity
from openbtk.pipelines import Pipeline

from .test_executor import _Rec

if TYPE_CHECKING:
    from collections.abc import Iterator

_LINES = "loader.general.pipeline_test_lines"
_UPPER = "preprocessor.general.pipeline_test_upper"
_WORDS = "chunker.general.pipeline_test_words"
_EXPLODE = "preprocessor.general.pipeline_test_explode"


class _Seen:
    """What each test double saw, cleared before every test."""

    log: ClassVar[dict[str, list[str]]] = {}
    pulled = 0
    max_lag: ClassVar[dict[str, int]] = {}
    consumed: ClassVar[dict[str, int]] = {}

    @classmethod
    def reset(cls) -> None:
        cls.log.clear()
        cls.max_lag.clear()
        cls.consumed.clear()
        cls.pulled = 0


@LOADER_REGISTRY.register("loader.general.dag_test_counting")
class _CountingLoader(BaseLoader[int, _Rec]):
    """Yields ``source`` records and counts how many have been pulled so far."""

    def load(self, source: int) -> Iterator[_Rec]:
        for i in range(source):
            _Seen.pulled += 1
            yield _Rec(record_id=str(i), text=f"r{i}")


class _Recording(BasePreprocessor[_Rec]):
    name = ""

    def process(self, record: _Rec) -> _Rec:
        _Seen.log.setdefault(self.name, []).append(record.text)
        count = _Seen.consumed.get(self.name, 0) + 1
        _Seen.consumed[self.name] = count
        lag = _Seen.pulled - count
        _Seen.max_lag[self.name] = max(_Seen.max_lag.get(self.name, 0), lag)
        return record


@PREPROCESSOR_REGISTRY.register("preprocessor.general.dag_test_left")
class _Left(_Recording):
    name = "left"


@PREPROCESSOR_REGISTRY.register("preprocessor.general.dag_test_right")
class _Right(_Recording):
    name = "right"


@PREPROCESSOR_REGISTRY.register("preprocessor.general.dag_test_merge")
class _Merge(_Recording):
    name = "merge"


@GUARDRAIL_REGISTRY.register("guardrail.general.dag_test_count")
class _CountingGuardrail(BaseGuardrail):
    checked = 0

    def check(self, payload: Any) -> GuardrailResult:
        type(self).checked += 1
        return GuardrailResult(
            passed=True,
            severity=GuardrailSeverity.INFO,
            guardrail_key=self.registry_key,
            message="ok",
        )


@pytest.fixture(autouse=True)
def _reset() -> None:
    _Seen.reset()
    _CountingGuardrail.checked = 0


def _step(step_id: str, type_: str, *after: str, **params: Any) -> StepConfig:
    return StepConfig(id=step_id, type=type_, params=params, after=list(after))


def _run(*steps: StepConfig) -> Any:
    return Pipeline.from_config(PipelineConfig(name="dag", steps=list(steps))).run()


def _counts(manifest: Any) -> dict[str, tuple[int, int]]:
    return {s.step_id: (s.records_in, s.records_out) for s in manifest.steps}


class TestFanOut:
    def test_every_branch_sees_every_record(self) -> None:
        manifest = _run(
            _step("load", _LINES, source=["a", "b", "c"]),
            _step("left", "preprocessor.general.dag_test_left", "load"),
            _step("right", "preprocessor.general.dag_test_right", "load"),
        )
        assert manifest.status == "success", manifest.error
        assert _Seen.log["left"] == ["a", "b", "c"]
        assert _Seen.log["right"] == ["a", "b", "c"]
        assert _counts(manifest) == {"load": (0, 3), "left": (3, 3), "right": (3, 3)}

    def test_the_branches_may_differ_in_kind(self) -> None:
        manifest = _run(
            _step("load", _LINES, source=["one two", "three"]),
            _step("upper", _UPPER, "load"),
            _step("words", _WORDS, "load"),
        )
        assert manifest.status == "success", manifest.error
        assert _counts(manifest)["upper"] == (2, 2)
        assert _counts(manifest)["words"] == (2, 3)  # a chunker fans records out

    def test_three_dependents(self) -> None:
        manifest = _run(
            _step("load", _LINES, source=["x"] * 4),
            _step("a", "preprocessor.general.dag_test_left", "load"),
            _step("b", "preprocessor.general.dag_test_right", "load"),
            _step("c", _UPPER, "load"),
        )
        assert manifest.status == "success"
        assert _counts(manifest)["c"] == (4, 4)
        assert _Seen.consumed == {"left": 4, "right": 4}

    def test_a_guardrail_on_the_fan_out_step_sees_each_record_once(self) -> None:
        pipeline = Pipeline.from_config(
            PipelineConfig(
                name="dag",
                steps=[
                    _step("load", _LINES, source=["a", "b", "c"]),
                    _step("left", "preprocessor.general.dag_test_left", "load"),
                    _step("right", "preprocessor.general.dag_test_right", "load"),
                ],
            )
        ).guard("guardrail.general.dag_test_count", at="after:load")
        manifest = pipeline.run()
        assert manifest.status == "success", manifest.error
        assert _CountingGuardrail.checked == 3  # not 6: it sits before the split
        (outcome,) = manifest.guardrail_outcomes
        assert outcome.checked_count == 3

    def test_a_guardrail_block_stops_every_branch(self) -> None:
        pipeline = Pipeline.from_config(
            PipelineConfig(
                name="dag",
                steps=[
                    _step("load", _LINES, source=["ok", "SECRET", "ok"]),
                    _step("left", "preprocessor.general.dag_test_left", "load"),
                    _step("right", "preprocessor.general.dag_test_right", "load"),
                ],
            )
        ).guard("guardrail.general.pipeline_test_reject_word", at="after:load")
        manifest = pipeline.run()
        assert manifest.status == "failed"
        assert "found SECRET" in (manifest.error or "")
        assert "SECRET" not in _Seen.log.get("left", []) + _Seen.log.get("right", [])


class TestFanIn:
    def test_a_step_reads_both_streams_interleaved(self) -> None:
        manifest = _run(
            _step("a", _LINES, source=["a0", "a1", "a2"]),
            _step("b", _LINES, source=["b0", "b1", "b2"]),
            _step("merge", "preprocessor.general.dag_test_merge", "a", "b"),
        )
        assert manifest.status == "success", manifest.error
        assert _Seen.log["merge"] == ["a0", "b0", "a1", "b1", "a2", "b2"]
        assert _counts(manifest)["merge"] == (6, 6)

    def test_unequal_streams_are_all_drained(self) -> None:
        _run(
            _step("a", _LINES, source=["a0"]),
            _step("b", _LINES, source=["b0", "b1", "b2"]),
            _step("merge", "preprocessor.general.dag_test_merge", "a", "b"),
        )
        assert _Seen.log["merge"] == ["a0", "b0", "b1", "b2"]

    def test_an_empty_stream_does_not_block_the_others(self) -> None:
        _run(
            _step("a", _LINES, source=[]),
            _step("b", _LINES, source=["b0", "b1"]),
            _step("merge", "preprocessor.general.dag_test_merge", "a", "b"),
        )
        assert _Seen.log["merge"] == ["b0", "b1"]

    def test_each_loader_is_digested_separately(self, tmp_path: Any) -> None:
        one, two = tmp_path / "one.txt", tmp_path / "two.txt"
        one.write_text("x\ny\n", encoding="utf-8")
        two.write_text("z\n", encoding="utf-8")
        manifest = _run(
            _step("a", "loader.general.pipeline_test_file_lines", source=str(one)),
            _step("b", "loader.general.pipeline_test_file_lines", source=str(two)),
            _step("merge", "preprocessor.general.dag_test_merge", "a", "b"),
        )
        assert manifest.status == "success", manifest.error
        assert {d.uri: d.record_count for d in manifest.input_digests} == {
            str(one): 2,
            str(two): 1,
        }

    def test_a_predecessor_named_twice_is_refused(self) -> None:
        manifest = _run(
            _step("a", _LINES, source=["x"]),
            _step("m", _UPPER, "a", "a"),
        )
        assert manifest.status == "failed"
        assert "more than once" in (manifest.error or "")


class TestDiamond:
    def test_the_merge_step_receives_each_record_by_both_paths(self) -> None:
        manifest = _run(
            _step("load", _LINES, source=["a", "b"]),
            _step("left", "preprocessor.general.dag_test_left", "load"),
            _step("right", "preprocessor.general.dag_test_right", "load"),
            _step("merge", "preprocessor.general.dag_test_merge", "left", "right"),
        )
        assert manifest.status == "success", manifest.error
        assert _counts(manifest) == {
            "load": (0, 2),
            "left": (2, 2),
            "right": (2, 2),
            "merge": (4, 4),
        }
        assert sorted(_Seen.log["merge"]) == ["a", "a", "b", "b"]


class TestSeveralLeaves:
    def test_independent_chains_all_run(self) -> None:
        manifest = _run(
            _step("a", _LINES, source=["x", "y"]),
            _step("b", _LINES, source=["z"]),
        )
        assert manifest.status == "success", manifest.error
        assert _counts(manifest) == {"a": (0, 2), "b": (0, 1)}

    def test_one_failing_chain_fails_the_run(self) -> None:
        manifest = _run(
            _step("a", _LINES, source=["fine"]),
            _step("b", _LINES, source=["BOOM"]),
            _step("boom", _EXPLODE, "b"),
        )
        assert manifest.status == "failed"
        assert {s.step_id: s.status for s in manifest.steps}["boom"] == "failed"


class TestFailureAttribution:
    def test_a_failure_in_one_branch_names_that_branch(self) -> None:
        manifest = _run(
            _step("load", _LINES, source=["ok", "BOOM", "ok"]),
            _step("good", "preprocessor.general.dag_test_left", "load"),
            _step("boom", _EXPLODE, "load"),
        )
        assert manifest.status == "failed"
        status = {s.step_id: s.status for s in manifest.steps}
        assert status["boom"] == "failed"
        assert status["good"] == "success" and status["load"] == "success"
        assert "boom" in (manifest.error or "")

    def test_a_failure_upstream_of_a_fan_out_is_not_blamed_on_a_branch(self) -> None:
        manifest = _run(
            _step("load", "loader.general.pipeline_test_always_fails", source=[]),
            _step("left", "preprocessor.general.dag_test_left", "load"),
            _step("right", "preprocessor.general.dag_test_right", "load"),
        )
        status = {s.step_id: s.status for s in manifest.steps}
        assert manifest.status == "failed"
        assert status["load"] == "failed"
        assert status["left"] == status["right"] == "success"


class TestMemory:
    def test_branches_advance_together_so_the_buffer_stays_small(self) -> None:
        """A sequential drain would hold the whole stream for the second branch (lag of
        the full 2000). Pulling the branches in turn keeps it at a record or two."""
        manifest = _run(
            _step("load", "loader.general.dag_test_counting", source=2000),
            _step("left", "preprocessor.general.dag_test_left", "load"),
            _step("right", "preprocessor.general.dag_test_right", "load"),
        )
        assert manifest.status == "success", manifest.error
        assert _Seen.consumed == {"left": 2000, "right": 2000}
        assert max(_Seen.max_lag.values()) <= 3

    def test_a_fan_in_does_not_buffer_a_stream_either(self) -> None:
        _run(
            _step("a", "loader.general.dag_test_counting", source=1000),
            _step("b", "loader.general.dag_test_counting", source=1000),
            _step("merge", "preprocessor.general.dag_test_merge", "a", "b"),
        )
        # both loaders bump the same counter; pulled - consumed stays tiny
        assert _Seen.consumed["merge"] == 2000
        assert _Seen.max_lag["merge"] <= 3


class TestValidation:
    def test_a_fan_out_pipeline_passes_validate(self) -> None:
        pipeline = Pipeline.from_config(
            PipelineConfig(
                name="dag",
                steps=[
                    _step("load", _LINES, source=[]),
                    _step("left", "preprocessor.general.dag_test_left", "load"),
                    _step("right", "preprocessor.general.dag_test_right", "load"),
                ],
            )
        )
        assert [i for i in pipeline.validate() if i.severity == "error"] == []

    def test_a_repeated_predecessor_is_reported_by_validate(self) -> None:
        pipeline = Pipeline.from_config(
            PipelineConfig(
                name="dag",
                steps=[_step("a", _LINES, source=[]), _step("m", _UPPER, "a", "a")],
            )
        )
        assert any("more than once" in i.message for i in pipeline.validate())

    def test_a_loader_with_a_predecessor_is_reported_by_validate(self) -> None:
        pipeline = Pipeline.from_config(
            PipelineConfig(
                name="dag",
                steps=[
                    _step("a", _LINES, source=[]),
                    _step("b", _LINES, "a", source=[]),
                ],
            )
        )
        assert any("root step" in i.message for i in pipeline.validate())

    def test_a_cycle_is_still_refused(self) -> None:
        pipeline = Pipeline.from_config(
            PipelineConfig(
                name="dag",
                steps=[_step("x", _UPPER, "y"), _step("y", _UPPER, "x")],
            )
        )
        assert any("cycle" in i.message.lower() for i in pipeline.validate())
