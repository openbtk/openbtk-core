"""Unit tests for openbtk.pipelines.executor, exercised through the public
Pipeline/Step API.

Registers small, deterministic test-double components once at module scope
(the same pattern tests/contract/conftest.py uses for its reference
implementations) rather than relying on real clinical_text components --
this suite is about the EXECUTOR's own DAG-walking, provenance and
guardrail logic, independent of what any one modality's components do. A
real end-to-end run of the actual clinical_text components is task 3.8's
own integration test.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel, ConfigDict

from openbtk.core.base import BaseChunker, BaseGuardrail, BaseLoader, BasePreprocessor
from openbtk.core.config import PipelineConfig, StepConfig
from openbtk.core.errors import LoaderError, ProcessingError
from openbtk.core.registry import (
    CHUNKER_REGISTRY,
    GUARDRAIL_REGISTRY,
    LOADER_REGISTRY,
    PREPROCESSOR_REGISTRY,
)
from openbtk.core.schemas import GuardrailResult, GuardrailSeverity
from openbtk.pipelines import Pipeline, Step
from openbtk.pipelines.executor import _is_secret_key

if TYPE_CHECKING:
    from collections.abc import Iterator


class _Rec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    record_id: str
    text: str


class _Chunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    chunk_id: str
    record_id: str
    text: str


@LOADER_REGISTRY.register("loader.general.pipeline_test_lines")
class _LineLoader(BaseLoader[list[str], _Rec]):
    """source is a plain list of strings, one record per entry -- no
    filesystem I/O needed for most executor tests."""

    def load(self, source: list[str]) -> Iterator[_Rec]:
        for i, line in enumerate(source):
            yield _Rec(record_id=str(i), text=line)


@LOADER_REGISTRY.register("loader.general.pipeline_test_file_lines")
class _FileLineLoader(BaseLoader[str, _Rec]):
    """source is a real file path -- for DataDigest-specific tests only."""

    def load(self, source: str) -> Iterator[_Rec]:
        with Path(source).open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                yield _Rec(record_id=str(i), text=line.rstrip("\n"))


@LOADER_REGISTRY.register("loader.general.pipeline_test_always_fails")
class _AlwaysFailingLoader(BaseLoader[list[str], _Rec]):
    """Raises unconditionally -- for testing that a loader step's OWN
    failure (not a downstream step's) is attributed correctly."""

    def load(self, source: list[str]) -> Iterator[_Rec]:
        raise LoaderError("boom", context={"count": len(source)})


@PREPROCESSOR_REGISTRY.register("preprocessor.general.pipeline_test_upper")
class _UpperPreprocessor(BasePreprocessor[_Rec]):
    def process(self, record: _Rec) -> _Rec:
        return _Rec(record_id=record.record_id, text=record.text.upper())


@PREPROCESSOR_REGISTRY.register("preprocessor.general.pipeline_test_explode")
class _ExplodingPreprocessor(BasePreprocessor[_Rec]):
    """Raises for any record whose text is exactly "BOOM" -- the
    controlled, reproducible failure this suite needs to test attribution."""

    def process(self, record: _Rec) -> _Rec:
        if record.text == "BOOM":
            raise ProcessingError("boom", context={"record_id": record.record_id})
        return record


@PREPROCESSOR_REGISTRY.register("preprocessor.general.pipeline_test_explode_plain")
class _PlainExceptionPreprocessor(BasePreprocessor[_Rec]):
    """Raises a bare ValueError, not an OpenBTKError -- proves the executor
    wraps an unexpected third-party/stdlib exception into a real
    ProcessingError rather than letting it propagate unattributed."""

    def process(self, record: _Rec) -> _Rec:
        if record.text == "BOOM":
            raise ValueError("not an OpenBTKError")
        return record


@CHUNKER_REGISTRY.register("chunker.general.pipeline_test_words")
class _WordChunker(BaseChunker[_Rec, _Chunk]):
    def chunk(self, record: _Rec) -> Iterator[_Chunk]:
        for i, word in enumerate(record.text.split()):
            yield _Chunk(
                chunk_id=f"{record.record_id}-{i}",
                record_id=record.record_id,
                text=word,
            )


@GUARDRAIL_REGISTRY.register("guardrail.general.pipeline_test_reject_word")
class _RejectWordGuardrail(BaseGuardrail):
    """BLOCKs any payload whose .text contains "SECRET"."""

    def check(self, payload: Any) -> GuardrailResult:
        text = getattr(payload, "text", str(payload))
        if "SECRET" in text:
            return GuardrailResult(
                passed=False,
                severity=GuardrailSeverity.BLOCK,
                guardrail_key=self.registry_key,
                message="found SECRET",
            )
        return GuardrailResult(
            passed=True,
            severity=GuardrailSeverity.INFO,
            guardrail_key=self.registry_key,
            message="ok",
        )


@GUARDRAIL_REGISTRY.register("guardrail.general.pipeline_test_warn_word")
class _WarnWordGuardrail(BaseGuardrail):
    """WARNING-severity (not BLOCK) failure for any payload whose .text
    contains "SUSPECT" -- the tally's WARNING branch has no other test."""

    def check(self, payload: Any) -> GuardrailResult:
        text = getattr(payload, "text", str(payload))
        if "SUSPECT" in text:
            return GuardrailResult(
                passed=False,
                severity=GuardrailSeverity.WARNING,
                guardrail_key=self.registry_key,
                message="found SUSPECT",
            )
        return GuardrailResult(
            passed=True,
            severity=GuardrailSeverity.INFO,
            guardrail_key=self.registry_key,
            message="ok",
        )


def _linear_pipeline(lines: list[str]) -> Pipeline:
    return (
        Pipeline("test")
        .add(Step("load", "loader.general.pipeline_test_lines", source=lines))
        .add(Step("upper", "preprocessor.general.pipeline_test_upper"))
        .add(Step("chunk", "chunker.general.pipeline_test_words"))
    )


class TestLinearPipelineSuccess:
    def test_status_is_success(self) -> None:
        manifest = _linear_pipeline(["one two", "three"]).run()
        assert manifest.status == "success"
        assert manifest.error is None

    def test_step_counts_are_accurate(self) -> None:
        manifest = _linear_pipeline(["one two", "three"]).run()
        by_id = {s.step_id: s for s in manifest.steps}
        assert by_id["load"].records_out == 2
        assert by_id["upper"].records_in == 2
        assert by_id["upper"].records_out == 2
        assert by_id["chunk"].records_in == 2
        assert by_id["chunk"].records_out == 3  # "one","two","three"

    def test_every_step_reports_success(self) -> None:
        manifest = _linear_pipeline(["one two", "three"]).run()
        assert all(s.status == "success" for s in manifest.steps)

    def test_component_provenance_is_populated(self) -> None:
        manifest = _linear_pipeline(["a"]).run()
        by_id = {s.step_id: s for s in manifest.steps}
        assert by_id["load"].component.class_name == "_LineLoader"
        assert (
            by_id["load"].component.registry_key == "loader.general.pipeline_test_lines"
        )

    def test_empty_source_yields_zero_records_not_an_error(self) -> None:
        manifest = _linear_pipeline([]).run()
        assert manifest.status == "success"
        by_id = {s.step_id: s for s in manifest.steps}
        assert by_id["load"].records_out == 0
        assert by_id["chunk"].records_out == 0


def _exploding_pipeline(lines: list[str]) -> Pipeline:
    return (
        Pipeline("test")
        .add(Step("load", "loader.general.pipeline_test_lines", source=lines))
        .add(Step("explode", "preprocessor.general.pipeline_test_explode"))
        .add(Step("chunk", "chunker.general.pipeline_test_words"))
    )


class TestFailureAttribution:
    def test_status_is_failed(self) -> None:
        manifest = _exploding_pipeline(["ok", "BOOM", "never reached"]).run()
        assert manifest.status == "failed"
        assert manifest.error is not None

    def test_only_the_raising_step_is_marked_failed(self) -> None:
        manifest = _exploding_pipeline(["ok", "BOOM"]).run()
        by_id = {s.step_id: s for s in manifest.steps}
        assert by_id["load"].status == "success"
        assert by_id["explode"].status == "failed"
        assert by_id["explode"].error is not None

    def test_partial_counts_are_real_not_fabricated(self) -> None:
        """ "BOOM" is the second of three lines: the loader yields both
        "ok" and "BOOM" (it never itself raises) before the preprocessor
        fails trying to transform "BOOM" -- the third line is never
        reached at all, and the preprocessor's own OUTPUT count reflects
        only the one record ("ok") it actually finished before failing."""
        manifest = _exploding_pipeline(["ok", "BOOM", "never reached"]).run()
        by_id = {s.step_id: s for s in manifest.steps}
        assert by_id["load"].records_out == 2
        assert by_id["explode"].records_in == 2
        assert by_id["explode"].records_out == 1

    def test_downstream_step_is_wired_but_never_receives_anything(self) -> None:
        """Wiring (building the lazy generator chain) happens for every
        declared step before any draining starts -- "chunk" DOES appear in
        the manifest, genuinely wired, but the failure upstream means it
        never actually pulls a single item."""
        manifest = _exploding_pipeline(["BOOM"]).run()
        by_id = {s.step_id: s for s in manifest.steps}
        assert "chunk" in by_id
        assert by_id["chunk"].status == "success"  # never itself raised
        assert by_id["chunk"].records_in == 0
        assert by_id["chunk"].records_out == 0

    def test_the_loader_itself_failing_is_attributed_to_the_loader(self) -> None:
        pipeline = (
            Pipeline("test")
            .add(
                Step("load", "loader.general.pipeline_test_always_fails", source=["a"])
            )
            .add(Step("upper", "preprocessor.general.pipeline_test_upper"))
        )
        manifest = pipeline.run()
        by_id = {s.step_id: s for s in manifest.steps}
        assert manifest.status == "failed"
        assert by_id["load"].status == "failed"
        assert by_id["load"].records_out == 0
        # "upper" was wired but the loader failed before yielding anything to it.
        assert by_id["upper"].status == "success"
        assert by_id["upper"].records_in == 0

    def test_a_non_openbtkerror_is_wrapped_and_still_attributed(self) -> None:
        pipeline = (
            Pipeline("test")
            .add(Step("load", "loader.general.pipeline_test_lines", source=["BOOM"]))
            .add(Step("explode", "preprocessor.general.pipeline_test_explode_plain"))
        )
        manifest = pipeline.run()
        by_id = {s.step_id: s for s in manifest.steps}
        assert manifest.status == "failed"
        assert by_id["explode"].status == "failed"
        assert "not an OpenBTKError" in (manifest.error or "")


class TestGuardrails:
    def test_block_halts_the_pipeline(self) -> None:
        pipeline = _linear_pipeline(["public text", "SECRET stuff"]).guard(
            "guardrail.general.pipeline_test_reject_word", at="after:upper"
        )
        manifest = pipeline.run()
        assert manifest.status == "failed"
        assert manifest.guardrail_outcomes[0].blocked_count == 1

    def test_warn_does_not_halt_but_still_tallies(self) -> None:
        pipeline = _linear_pipeline(["public text", "SECRET stuff"]).guard(
            "guardrail.general.pipeline_test_reject_word",
            at="after:upper",
            on_violation="warn",
        )
        manifest = pipeline.run()
        assert manifest.status == "success"
        outcome = manifest.guardrail_outcomes[0]
        assert outcome.blocked_count == 1
        assert outcome.checked_count == 2

    def test_no_violation_reports_zero_blocked(self) -> None:
        pipeline = _linear_pipeline(["clean one", "clean two"]).guard(
            "guardrail.general.pipeline_test_reject_word", at="after:upper"
        )
        manifest = pipeline.run()
        assert manifest.status == "success"
        outcome = manifest.guardrail_outcomes[0]
        assert outcome.checked_count == 2
        assert outcome.blocked_count == 0

    def test_unknown_attachment_point_fails_the_run(self) -> None:
        pipeline = _linear_pipeline(["a"]).guard(
            "guardrail.general.pipeline_test_reject_word", at="after:nonexistent_step"
        )
        manifest = pipeline.run()
        assert manifest.status == "failed"

    def test_warning_severity_does_not_halt_and_tallies_separately_from_blocked(
        self,
    ) -> None:
        pipeline = _linear_pipeline(["clean", "SUSPECT text"]).guard(
            "guardrail.general.pipeline_test_warn_word", at="after:upper"
        )
        manifest = pipeline.run()
        assert manifest.status == "success"
        outcome = manifest.guardrail_outcomes[0]
        assert outcome.warned_count == 1
        assert outcome.blocked_count == 0
        assert outcome.checked_count == 2

    def test_sample_messages_are_capped_not_one_per_record(self) -> None:
        lines = [f"SUSPECT {i}" for i in range(10)]
        pipeline = _linear_pipeline(lines).guard(
            "guardrail.general.pipeline_test_warn_word", at="after:upper"
        )
        manifest = pipeline.run()
        outcome = manifest.guardrail_outcomes[0]
        assert outcome.warned_count == 10
        assert len(outcome.sample_messages) == 5


class TestUnsupportedShapes:
    """What is still refused. (Fan-out, fan-in and several leaves are supported now;
    see ``test_dag.py``.)"""

    def test_loader_step_with_a_predecessor_fails_the_run(self) -> None:
        config = PipelineConfig(
            name="bad",
            steps=[
                StepConfig(
                    id="a",
                    type="loader.general.pipeline_test_lines",
                    params={"source": []},
                ),
                StepConfig(
                    id="b",
                    type="loader.general.pipeline_test_lines",
                    params={"source": []},
                    after=["a"],
                ),
            ],
        )
        manifest = Pipeline.from_config(config).run()
        assert manifest.status == "failed"
        assert "root step" in (manifest.error or "")

    def test_preprocessor_as_root_fails_the_run(self) -> None:
        config = PipelineConfig(
            name="bad",
            steps=[StepConfig(id="a", type="preprocessor.general.pipeline_test_upper")],
        )
        manifest = Pipeline.from_config(config).run()
        assert manifest.status == "failed"
        assert "predecessor" in (manifest.error or "")

    def test_chunker_as_root_fails_the_run(self) -> None:
        config = PipelineConfig(
            name="bad",
            steps=[StepConfig(id="a", type="chunker.general.pipeline_test_words")],
        )
        manifest = Pipeline.from_config(config).run()
        assert manifest.status == "failed"
        assert "predecessor" in (manifest.error or "")

    def test_non_executable_category_fails_the_run(self) -> None:
        """A guardrail-category component used as a STEP (not a guard()
        attachment) has no dispatch entry -- this must fail loudly, not
        silently no-op."""
        config = PipelineConfig(
            name="bad",
            steps=[
                StepConfig(id="a", type="guardrail.general.pipeline_test_reject_word")
            ],
        )
        manifest = Pipeline.from_config(config).run()
        assert manifest.status == "failed"
        assert "not yet executable" in (manifest.error or "")

    def test_unregistered_type_fails_the_run(self) -> None:
        config = PipelineConfig(
            name="bad",
            steps=[StepConfig(id="a", type="loader.general.does_not_exist")],
        )
        manifest = Pipeline.from_config(config).run()
        assert manifest.status == "failed"


# A synthetic, unassigned, human-readable placeholder for a real API key --
# never a real credential -- used only to prove the executor's secret
# redaction actually fires.
_FAKE_API_KEY = "sk-real-secret-value"  # pragma: allowlist secret


class TestSecretRedaction:
    def test_a_credential_shaped_param_is_redacted_in_the_manifest(self) -> None:
        pipeline = Pipeline("test").add(
            Step(
                "load",
                "loader.general.pipeline_test_lines",
                source=[],
                api_key=_FAKE_API_KEY,
            )
        )
        manifest = pipeline.run()
        assert _FAKE_API_KEY not in str(manifest.config)
        steps_list = manifest.config["steps"]
        assert isinstance(steps_list, list)
        first_step = steps_list[0]
        assert isinstance(first_step, dict)
        params = first_step["params"]
        assert isinstance(params, dict)
        assert params["api_key"] == "[REDACTED]"

    def test_a_token_count_is_a_setting_not_a_secret(self) -> None:
        """Regression: ``max_tokens`` matched the credential pattern, so the
        manifest recorded ``"[REDACTED]"`` for a chunker's size limit and the
        run could not be replayed."""
        manifest = (
            Pipeline("test")
            .add(
                Step(
                    "load",
                    "loader.general.pipeline_test_lines",
                    source=[],
                    max_tokens=200,
                    access_token=_FAKE_API_KEY,
                )
            )
            .run()
        )
        steps_list = manifest.config["steps"]
        assert isinstance(steps_list, list) and isinstance(steps_list[0], dict)
        params = steps_list[0]["params"]
        assert isinstance(params, dict)
        assert params["max_tokens"] == 200
        assert params["access_token"] == "[REDACTED]"

    @pytest.mark.parametrize(
        "key",
        [
            "api_key",
            "API_KEY",
            "access_token",
            "hf_token",
            "auth_token",
            "token",
            "client_secret",
            "password",
            "db_password",
            "credentials",
            "tokenizer_key",
        ],
    )
    def test_credential_shaped_keys_are_secret(self, key: str) -> None:
        assert _is_secret_key(key)

    @pytest.mark.parametrize(
        "key",
        [
            "max_tokens",
            "overlap_tokens",
            "Max_Tokens",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "num_tokens",
            "count_tokens",
            "mode",
        ],
    )
    def test_length_and_ordinary_keys_are_not(self, key: str) -> None:
        assert not _is_secret_key(key)


class TestDataDigest:
    def test_real_file_source_gets_a_real_sha256(self, tmp_path: Any) -> None:
        note = tmp_path / "notes.txt"
        note.write_text("line one\nline two\n", encoding="utf-8")
        pipeline = Pipeline("test").add(
            Step("load", "loader.general.pipeline_test_file_lines", source=str(note))
        )
        manifest = pipeline.run()
        assert manifest.status == "success"
        assert len(manifest.input_digests) == 1
        digest = manifest.input_digests[0]
        assert digest.uri == str(note)
        assert digest.sha256 == hashlib.sha256(note.read_bytes()).hexdigest()
        assert digest.record_count == 2

    def test_non_file_source_has_no_sha256_but_is_not_fabricated(self) -> None:
        manifest = (
            Pipeline("test")
            .add(Step("load", "loader.general.pipeline_test_lines", source=["a", "b"]))
            .run()
        )
        assert manifest.input_digests == []  # list[str] source isn't a path at all

    def test_a_directory_source_has_no_sha256_but_is_not_fabricated(
        self, tmp_path: Any
    ) -> None:
        """A directory is a real, existing str source -- but not a single
        file to hash. See _digest_source's own docstring."""
        manifest = (
            Pipeline("test")
            .add(
                Step(
                    "load",
                    "loader.general.pipeline_test_file_lines",
                    source=str(tmp_path),
                )
            )
            .run()
        )
        assert manifest.status == "failed"  # not a file at all -- open() itself fails
        assert len(manifest.input_digests) == 1
        assert manifest.input_digests[0].sha256 is None

    def test_path_kwarg_is_accepted_as_a_fallback_for_source(
        self, tmp_path: Any
    ) -> None:
        """docs/03_ARCHITECTURE.md section 7.3's own YAML example names the
        loader's call argument "path", not "source" -- both spellings must
        resolve to the same thing."""
        note = tmp_path / "notes.txt"
        note.write_text("only line\n", encoding="utf-8")
        manifest = (
            Pipeline("test")
            .add(
                Step("load", "loader.general.pipeline_test_file_lines", path=str(note))
            )
            .run()
        )
        assert manifest.status == "success"
        assert manifest.input_digests[0].record_count == 1


class TestManifestShape:
    def test_run_id_is_unique_per_run(self) -> None:
        pipeline = _linear_pipeline(["a"])
        first = pipeline.run()
        second = pipeline.run()
        assert first.run_id != second.run_id

    def test_config_snapshot_matches_the_built_pipeline(self) -> None:
        manifest = _linear_pipeline(["a"]).run()
        assert manifest.config["name"] == "test"
        steps_list = manifest.config["steps"]
        assert isinstance(steps_list, list)
        assert len(steps_list) == 3
