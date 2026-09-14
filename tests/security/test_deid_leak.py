"""Release blocker (docs/07_TEST_CHARTER.md section 3.5): no PHI from the
labelled corpus may survive into a DeidReport or de-identified output text.

Two genuinely different claims, both worth checking separately:

  * ``test_no_phi_in_deid_report`` is the test charter's own named example
    (``def test_no_phi_in_deid_report(labelled_phi_corpus): ...``). It
    passes for every category, including NAME and GEOGRAPHIC_SUBDIVISION,
    which RuleRecognizer never detects -- and that's fine: ``Detection``
    structurally cannot hold a matched value (openbtk.deid.schemas'
    docstring), so a category having zero detections doesn't create a
    reportable leak, it just means the report says nothing about it.

  * ``test_no_phi_in_deidentified_output_text`` is the operationally
    sharper question: does the actual redacted DOCUMENT still contain PHI?
    For a category no recognizer attempts, the honest answer today is
    yes -- an undetected name or address is never replaced, so it survives
    verbatim in ``result.text``. That is real, tracked via
    ``TestKnownTextLeakForUndetectedCategories`` as an
    ``xfail(strict=True)``, not silently passed over: the moment NER
    (task 2.4) starts catching these, this test flips to unexpectedly
    passing and forces an update here.

``test_no_phi_in_run_manifest`` from the same charter section is still not
buildable -- ``RunManifest`` remains deferred (core/provenance.py's own
module docstring) and no pipeline executor exists yet (M3+).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from openbtk.deid.engine import DeidEngine
from openbtk.deid.schemas import PHICategory

if TYPE_CHECKING:
    from fixtures.labelled_phi_corpus import LabelledPHICorpus

_NER_ONLY_CATEGORIES = {PHICategory.NAME, PHICategory.GEOGRAPHIC_SUBDIVISION}


def test_no_phi_in_deid_report(labelled_phi_corpus: LabelledPHICorpus) -> None:
    engine = DeidEngine(recognizers=["rule"])
    for doc in labelled_phi_corpus.documents:
        result = engine.deidentify(doc.text, patient_id=doc.document_id)
        blob = result.report.model_dump_json()
        for span in doc.spans:
            assert span.value not in blob, (
                f"{span.category.value} value leaked into DeidReport for "
                f"{doc.document_id}"
            )


class TestNoTextLeakForRuleCoverableCategories:
    """The real, must-pass claim: every category RuleRecognizer actually
    attempts must not survive into the de-identified output text."""

    def test_no_rule_coverable_identifier_survives_deidentification(
        self, labelled_phi_corpus: LabelledPHICorpus
    ) -> None:
        engine = DeidEngine(recognizers=["rule"])
        for doc in labelled_phi_corpus.documents:
            result = engine.deidentify(doc.text, patient_id=doc.document_id)
            for span in doc.spans:
                if span.category in _NER_ONLY_CATEGORIES:
                    continue
                assert span.value not in result.text, (
                    f"{span.category.value} value leaked into de-identified "
                    f"text for {doc.document_id}"
                )


class TestKnownTextLeakForUndetectedCategories:
    """Honest, tracked gap: NAME and GEOGRAPHIC_SUBDIVISION are never
    detected by RuleRecognizer, so they are never replaced, so they DO
    survive into the de-identified text today. xfail(strict=True) so this
    turns into a hard failure -- forcing this file to be updated -- the
    moment NERRecognizer (task 2.4) starts catching them."""

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Known M2 gap: RuleRecognizer does not attempt NAME or "
            "GEOGRAPHIC_SUBDIVISION (see its own module docstring; that is "
            "NERRecognizer's job, ADR-0006 task 2.4, not yet built), so "
            "these values are never replaced and survive verbatim in the "
            "de-identified text. When NER lands and starts catching them, "
            "this test will start passing unexpectedly (strict=True "
            "catches that) -- fold this check into "
            "TestNoTextLeakForRuleCoverableCategories at that point rather "
            "than leaving a stale xfail."
        ),
    )
    def test_ner_only_categories_do_not_leak_into_deidentified_text(
        self, labelled_phi_corpus: LabelledPHICorpus
    ) -> None:
        engine = DeidEngine(recognizers=["rule"])
        for doc in labelled_phi_corpus.documents:
            result = engine.deidentify(doc.text, patient_id=doc.document_id)
            for span in doc.spans:
                if span.category not in _NER_ONLY_CATEGORIES:
                    continue
                assert span.value not in result.text
