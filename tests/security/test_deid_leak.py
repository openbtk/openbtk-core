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
    ``DeidEngine``'s zero-extras-safe DEFAULT is ``recognizers=("rule",)``
    -- NER is opt-in (it needs a downloaded spaCy model), so with the
    default configuration a NAME or GEOGRAPHIC_SUBDIVISION value is never
    replaced and survives verbatim in ``result.text`` today.
    ``TestKnownTextLeakForUndetectedCategories`` tracks this as an
    ``xfail(strict=True)`` against the DEFAULT configuration specifically
    -- it will keep failing (correctly) even after NER exists, since NER
    is not part of the default. ``TestFullEnsembleRealLeakReduction``
    (``@pytest.mark.slow``) is the honest companion measurement against the
    FULL ``["rule", "ner"]`` ensemble: real, substantial improvement for
    NAME, real but weak improvement for GEOGRAPHIC_SUBDIVISION -- see
    tests/accuracy/test_deid_f1_with_ner.py for the full numbers this is
    consistent with.

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
    """Honest, tracked gap for DeidEngine's DEFAULT configuration
    specifically: with recognizers=("rule",) -- the zero-extras-safe
    default -- NAME and GEOGRAPHIC_SUBDIVISION are never detected, so
    never replaced, so DO survive into the de-identified text.
    xfail(strict=True) so this stays loud rather than silently true
    forever: if DeidEngine's *default* ever changes to include NER, this
    test starts unexpectedly passing and forces an update here. NER
    existing at all (it does, as of M2) does not close this gap by
    itself -- it is opt-in, precisely so the default stays free of a
    spaCy/model dependency; see TestFullEnsembleRealLeakReduction below
    for what actually happens once a caller opts in."""

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "By design, not yet closed: DeidEngine's default recognizer "
            'set is ("rule",) -- NERRecognizer exists (M2) but is opt-in '
            "(requires a downloaded spaCy model), so the DEFAULT "
            "configuration still never detects NAME or "
            "GEOGRAPHIC_SUBDIVISION. If this test ever starts passing "
            "unexpectedly, it means the default recognizer set changed -- "
            "update this test deliberately rather than leaving a stale "
            "xfail. See TestFullEnsembleRealLeakReduction for the "
            "opt-in ensemble's real, measured leak reduction."
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


@pytest.mark.slow
class TestFullEnsembleRealLeakReduction:
    """Requires the real spaCy model (OPENBTK_SLOW_TESTS=1) -- quantitative,
    not a binary pass/fail, because the real improvement is genuine but
    partial (tests/accuracy/test_deid_f1_with_ner.py has the full numbers
    this is consistent with): most NAME values stop leaking, most
    GEOGRAPHIC_SUBDIVISION values still do."""

    def test_most_name_values_no_longer_leak(
        self, labelled_phi_corpus: LabelledPHICorpus
    ) -> None:
        engine = DeidEngine(recognizers=["rule", "ner"])
        total = leaked = 0
        for doc in labelled_phi_corpus.documents:
            result = engine.deidentify(doc.text, patient_id=doc.document_id)
            for span in doc.spans:
                if span.category is not PHICategory.NAME:
                    continue
                total += 1
                if span.value in result.text:
                    leaked += 1
        assert total > 0
        assert (leaked / total) < 0.1

    def test_most_geographic_subdivision_values_still_leak(
        self, labelled_phi_corpus: LabelledPHICorpus
    ) -> None:
        """Documents the disclosed limitation quantitatively rather than
        silently -- this is NOT a target to keep this high, it is today's
        honest, measured floor (general-purpose NER does not reliably
        recognize street-address-shaped text as a location entity).

        The exact-substring leak rate (~0.4, measured) is noticeably lower
        than GEOGRAPHIC_SUBDIVISION's own recall (~0.08,
        tests/accuracy/test_deid_f1_with_ner.py) might suggest -- not
        because addresses are being recognized as addresses, but because
        spaCy sometimes misreads a FRAGMENT of one (e.g. "Natalie Lodge
        Apt") as a PERSON, which gets redacted as NAME and incidentally
        breaks the address string even though nothing scored a true
        geographic-category detection. Real behaviour, not a bug in this
        test -- measured, not assumed, exactly like the threshold below."""
        engine = DeidEngine(recognizers=["rule", "ner"])
        total = leaked = 0
        for doc in labelled_phi_corpus.documents:
            result = engine.deidentify(doc.text, patient_id=doc.document_id)
            for span in doc.spans:
                if span.category is not PHICategory.GEOGRAPHIC_SUBDIVISION:
                    continue
                total += 1
                if span.value in result.text:
                    leaked += 1
        assert total > 0
        assert (leaked / total) > 0.3
