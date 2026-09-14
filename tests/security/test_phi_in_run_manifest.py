"""Adversarial PHI-safety test, named directly in docs/07_TEST_CHARTER.md
section 3.5 (a **release blocker**): every value from a labelled synthetic
PHI corpus, run through the real end-to-end clinical_text pipeline, must
never appear anywhere in the serialised ``RunManifest``.

tests/security/conftest.py's own docstring deferred exactly this test at
M1: "RunManifest is core/provenance.py's deferred second increment;
DeidReport and the pipeline executor belong to M2/M3" -- all three now
exist (tasks 3.6/3.7), closing that gap.

**Why this test is expected to pass even though the default recognizer
set does not fully de-identify the corpus** (``DeidPreprocessor``'s
default ``recognizers=("rule",)`` never detects ``NAME`` or
``GEOGRAPHIC_SUBDIVISION`` -- see tests/security/test_deid_leak.py's own
xfail for that already-disclosed gap): this test is not really about
de-identification completeness at all. ``RunManifest``, ``StepProvenance``
and ``DataDigest`` structurally never carry record or chunk TEXT anywhere
-- only counts, a hashed/plain source path, and component identity. So
this test is a real, permanent regression guard against a *future* change
(e.g. attaching a sample record to a manifest for debugging, or a
guardrail whose ``message``/``details`` echoes matched text) rather than
evidence that today's de-identification is complete -- that claim is
tests/accuracy/'s job, not this file's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from openbtk.data import clinical_text as _clinical_text  # noqa: F401
from openbtk.pipelines import Pipeline, Step

if TYPE_CHECKING:
    from pathlib import Path

    from fixtures.labelled_phi_corpus import LabelledPHICorpus
    from openbtk.core.provenance import RunManifest


def _run_full_pipeline(corpus: LabelledPHICorpus, tmp_path: Path) -> RunManifest:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    for doc in corpus.documents:
        (notes_dir / f"{doc.document_id}.txt").write_text(doc.text, encoding="utf-8")
    pipeline = (
        Pipeline("phi-leak-check")
        .add(Step("load", "loader.clinical_text.plain_text", path=str(notes_dir)))
        .add(Step("deid", "preprocessor.general.deidentify", mode="redact"))
        .add(Step("segment", "preprocessor.clinical_text.section_segment"))
        .add(Step("chunk", "chunker.clinical_text.section_aware", max_tokens=100))
    )
    return pipeline.run()


def test_no_phi_in_run_manifest(
    labelled_phi_corpus: LabelledPHICorpus, tmp_path: Path
) -> None:
    manifest = _run_full_pipeline(labelled_phi_corpus, tmp_path)
    assert manifest.status == "success", manifest.error
    blob = manifest.model_dump_json()
    for identifier in labelled_phi_corpus.all_identifiers:
        assert identifier not in blob


def test_manifest_records_a_real_hash_not_document_content(
    labelled_phi_corpus: LabelledPHICorpus, tmp_path: Path
) -> None:
    """The one place a manifest DOES reference the input at all --
    DataDigest -- must point at the directory path, never at anything
    derived from document content."""
    manifest = _run_full_pipeline(labelled_phi_corpus, tmp_path)
    assert manifest.status == "success", manifest.error
    assert len(manifest.input_digests) == 1
    digest = manifest.input_digests[0]
    assert digest.uri == str(tmp_path / "notes")
    assert digest.record_count == len(labelled_phi_corpus.documents)
