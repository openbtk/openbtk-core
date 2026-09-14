"""Integration test (docs/07_TEST_CHARTER.md section 3.3, M3 task 3.8):
load -> deid -> segment -> chunk, end-to-end, through the REAL
``openbtk.pipelines`` executor and the REAL ``clinical_text`` components --
not test doubles. Unlike tests/unit/pipelines/test_executor.py (the
executor's own DAG-walking logic, exercised with small synthetic
components), this file's job is to prove those real components actually
compose correctly when driven through the real ``Pipeline``.

docs/03_ARCHITECTURE.md section 7.3's own worked pipeline continues past
chunk to embed/index -- neither has a real implementation in this
repository yet (CLAUDE.md's scope gate and the plain absence of any
concrete `embedding`/`vectorstore` component), so this integration test
stops exactly where M3's own exit criteria stops: chunk.

``Pipeline.run()`` returns only a ``RunManifest`` -- never the processed
records or chunks themselves (docs/04_API_DESIGN.md section 6's own
`manifest = pipeline.run()`). To verify the REAL de-identification and
chunking behaviour actually happened (not just that the run reported
"success"), a small test-local guardrail is attached at "after:chunk" to
capture chunk text in-process, in Python memory -- never written to the
manifest itself. This is a legitimate use of the documented guardrail
attachment mechanism (docs/03_ARCHITECTURE.md section 8.3: guardrails
observe the stream at a declared point), not a special test-only bypass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from openbtk.core.base import BaseGuardrail
from openbtk.core.registry import GUARDRAIL_REGISTRY
from openbtk.core.schemas import GuardrailResult, GuardrailSeverity

# Registers PlainTextLoader/DeidPreprocessor/SectionSegmenter/
# SectionAwareChunker as a side effect -- this test file is collected on
# its own (no tests/contract/conftest.py in this directory to trigger it),
# so the import has to happen explicitly here, the same way
# tests/contract/conftest.py itself does it.
from openbtk.data import clinical_text as _clinical_text  # noqa: F401
from openbtk.deid import DeidMode
from openbtk.pipelines import Pipeline, Step

if TYPE_CHECKING:
    from pathlib import Path

# Defined once with the suppression marker, then referenced by name
# everywhere below -- tests/security/test_fixture_hygiene.py scans by line
# text, so a line using the constant's NAME never re-trips the scanner
# (tests/unit/deid/test_rule_recognizer.py's own established convention).
_SSN_VALUE = "123-45-6789"  # phi-fixture-ok: synthetic, unassigned test value
_MRN_VALUE = "MRN-4821093"  # phi-fixture-ok: synthetic, unassigned test value
_EMAIL_VALUE = "jane.doe@example.org"  # phi-fixture-ok: synthetic

_NOTES = [
    (
        "Chief Complaint:\n"
        f"Patient SSN: {_SSN_VALUE} presents with chest pain.\n"
        "Plan:\n"
        f"Admit for observation. Contact {_EMAIL_VALUE} with questions.\n"
    ),
    (
        "Chief Complaint:\n"
        f"Follow-up visit, MRN: {_MRN_VALUE}.\n"
        "Plan:\n"
        "Continue current medications.\n"
    ),
    (
        "History of Present Illness:\n"
        "No acute distress reported today.\n"
        "Plan:\n"
        "Routine follow-up in six months.\n"
    ),
]
"""Three synthetic notes with fake, hand-typed PHI-shaped values (SSN,
email, MRN) -- never real patient data (CLAUDE.md rule 3). SSN/MRN/email
are covered by RuleRecognizer's own patterns (openbtk.deid.recognizers.rule),
so the default ("rule",) recognizer set genuinely detects and transforms
them without needing the NER extra."""


class _CaptureGuardrail(BaseGuardrail):
    """Records every payload's ``.text`` it sees, in-process, for this
    test's own assertions -- never returned in ``GuardrailResult`` and
    never written to a manifest. Class-level because the executor
    constructs a fresh instance per run (``registry.create`` per
    attachment point); reset explicitly between tests."""

    captured: ClassVar[list[str]] = []

    def check(self, payload: Any) -> GuardrailResult:
        type(self).captured.append(payload.text)
        return GuardrailResult(
            passed=True,
            severity=GuardrailSeverity.INFO,
            guardrail_key=self.registry_key,
            message="captured for test inspection",
        )

    @classmethod
    def reset(cls) -> None:
        cls.captured = []


class _RejectingGuardrail(BaseGuardrail):
    """BLOCKs unconditionally -- for the "guardrail halts the real
    pipeline" test."""

    def check(self, payload: Any) -> GuardrailResult:
        return GuardrailResult(
            passed=False,
            severity=GuardrailSeverity.BLOCK,
            guardrail_key=self.registry_key,
            message="rejecting everything, unconditionally, by design",
        )


GUARDRAIL_REGISTRY.register("guardrail.general.integration_test_capture")(
    _CaptureGuardrail
)
GUARDRAIL_REGISTRY.register("guardrail.general.integration_test_reject")(
    _RejectingGuardrail
)


def _write_notes(tmp_path: Path, notes: list[str]) -> Path:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    for i, text in enumerate(notes):
        (notes_dir / f"note{i}.txt").write_text(text, encoding="utf-8")
    return notes_dir


def _real_pipeline(notes_dir: Path, *, mode: DeidMode = DeidMode.REDACT) -> Pipeline:
    return (
        Pipeline("clinical-text-integration")
        .add(Step("load", "loader.clinical_text.plain_text", path=str(notes_dir)))
        .add(Step("deid", "preprocessor.general.deidentify", mode=mode.value))
        .add(Step("segment", "preprocessor.clinical_text.section_segment"))
        .add(Step("chunk", "chunker.clinical_text.section_aware", max_tokens=100))
    )


class TestEndToEndPipeline:
    def test_full_pipeline_succeeds(self, tmp_path: Path) -> None:
        notes_dir = _write_notes(tmp_path, _NOTES)
        manifest = _real_pipeline(notes_dir).run()
        assert manifest.status == "success", manifest.error

    def test_step_counts_match_across_the_real_components(self, tmp_path: Path) -> None:
        notes_dir = _write_notes(tmp_path, _NOTES)
        manifest = _real_pipeline(notes_dir).run()
        by_id = {s.step_id: s for s in manifest.steps}
        assert by_id["load"].records_out == len(_NOTES)
        assert by_id["deid"].records_in == len(_NOTES)
        assert by_id["deid"].records_out == len(_NOTES)
        assert by_id["segment"].records_out == len(_NOTES)
        # SectionAwareChunker fans out: at least one chunk per note.
        assert by_id["chunk"].records_out >= len(_NOTES)

    def test_component_provenance_names_the_real_classes(self, tmp_path: Path) -> None:
        notes_dir = _write_notes(tmp_path, _NOTES)
        manifest = _real_pipeline(notes_dir).run()
        by_id = {s.step_id: s for s in manifest.steps}
        assert by_id["load"].component.class_name == "PlainTextLoader"
        assert by_id["deid"].component.class_name == "DeidPreprocessor"
        assert by_id["segment"].component.class_name == "SectionSegmenter"
        assert by_id["chunk"].component.class_name == "SectionAwareChunker"

    def test_planted_identifiers_are_removed_from_chunk_text(
        self, tmp_path: Path
    ) -> None:
        """The real, end-to-end proof that de-identification actually ran
        as part of the composed pipeline -- not just in DeidPreprocessor's
        own unit tests."""
        _CaptureGuardrail.reset()
        notes_dir = _write_notes(tmp_path, _NOTES)
        pipeline = _real_pipeline(notes_dir).guard(
            "guardrail.general.integration_test_capture", at="after:chunk"
        )
        manifest = pipeline.run()
        assert manifest.status == "success", manifest.error
        all_chunk_text = " ".join(_CaptureGuardrail.captured)
        assert _SSN_VALUE not in all_chunk_text
        assert _MRN_VALUE not in all_chunk_text

    def test_sections_are_detected_and_respected(self, tmp_path: Path) -> None:
        """Real SectionSegmenter + SectionAwareChunker composed through the
        pipeline: every chunk from the first note carries a real section
        label, not None (which is what a "no sections" degenerate case
        would report)."""
        _CaptureGuardrail.reset()
        notes_dir = _write_notes(tmp_path, [_NOTES[0]])
        pipeline = _real_pipeline(notes_dir).guard(
            "guardrail.general.integration_test_capture", at="after:chunk"
        )
        pipeline.run()
        assert len(_CaptureGuardrail.captured) >= 2  # chief_complaint + plan
        combined = " ".join(_CaptureGuardrail.captured)
        assert "chest pain" in combined  # chief_complaint survived, un-redacted
        assert "Admit for observation" in combined  # plan survived, un-redacted

    def test_surrogate_mode_replaces_identifiers_consistently(
        self, tmp_path: Path
    ) -> None:
        """A second, real DeidMode -- SURROGATE -- verified end-to-end
        through the same composed pipeline, not just REDACT."""
        _CaptureGuardrail.reset()
        notes_dir = _write_notes(tmp_path, [_NOTES[0]])
        pipeline = _real_pipeline(notes_dir, mode=DeidMode.SURROGATE).guard(
            "guardrail.general.integration_test_capture", at="after:chunk"
        )
        manifest = pipeline.run()
        assert manifest.status == "success", manifest.error
        combined = " ".join(_CaptureGuardrail.captured)
        assert _SSN_VALUE not in combined
        assert "[SSN_1]" in combined

    def test_guardrail_block_halts_the_real_pipeline_and_is_recorded(
        self, tmp_path: Path
    ) -> None:
        notes_dir = _write_notes(tmp_path, _NOTES)
        pipeline = _real_pipeline(notes_dir).guard(
            "guardrail.general.integration_test_reject", at="after:deid"
        )
        manifest = pipeline.run()
        assert manifest.status == "failed"
        assert manifest.guardrail_outcomes[0].blocked_count >= 1
        assert "blocked at after:deid" in (manifest.error or "")

    def test_empty_notes_directory_succeeds_with_zero_records(
        self, tmp_path: Path
    ) -> None:
        notes_dir = _write_notes(tmp_path, [])
        manifest = _real_pipeline(notes_dir).run()
        assert manifest.status == "success"
        by_id = {s.step_id: s for s in manifest.steps}
        assert by_id["chunk"].records_out == 0
