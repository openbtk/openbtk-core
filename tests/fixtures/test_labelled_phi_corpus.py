"""Self-tests for the labelled PHI corpus generator itself.

If this fixture's ground-truth offsets are wrong, every accuracy number
computed against it downstream is meaningless -- these tests exist to make
that failure mode impossible to miss.
"""

from __future__ import annotations

import itertools

from openbtk.deid.schemas import PHICategory

from .labelled_phi_corpus import (
    DEFAULT_SEED,
    LabelledPHICorpus,
    build_labelled_phi_corpus,
)

_NON_TEXT_CATEGORIES = {PHICategory.FULL_FACE_PHOTO, PHICategory.BIOMETRIC_IDENTIFIER}
_TEXT_REPRESENTABLE_CATEGORIES = {
    c for c in PHICategory if c not in _NON_TEXT_CATEGORIES
}


class TestOffsetsAreCorrect:
    def test_every_span_value_matches_the_text_slice_exactly(self) -> None:
        corpus = build_labelled_phi_corpus(n_documents=5)
        for doc in corpus.documents:
            for span in doc.spans:
                assert doc.text[span.start : span.end] == span.value

    def test_spans_do_not_overlap_within_a_document(self) -> None:
        corpus = build_labelled_phi_corpus(n_documents=5)
        for doc in corpus.documents:
            ordered = sorted(doc.spans, key=lambda s: s.start)
            for a, b in itertools.pairwise(ordered):
                assert a.end <= b.start


class TestCoverage:
    def test_every_text_representable_category_appears(self) -> None:
        corpus = build_labelled_phi_corpus(n_documents=3)
        assert corpus.categories_present == _TEXT_REPRESENTABLE_CATEGORIES

    def test_non_text_categories_never_appear(self) -> None:
        corpus = build_labelled_phi_corpus(n_documents=3)
        assert PHICategory.FULL_FACE_PHOTO not in corpus.categories_present
        assert PHICategory.BIOMETRIC_IDENTIFIER not in corpus.categories_present

    def test_document_count_matches_request(self) -> None:
        corpus = build_labelled_phi_corpus(n_documents=7)
        assert len(corpus.documents) == 7

    def test_document_ids_are_unique(self) -> None:
        corpus = build_labelled_phi_corpus(n_documents=10)
        ids = [doc.document_id for doc in corpus.documents]
        assert len(ids) == len(set(ids))


class TestDeterminism:
    def test_same_seed_produces_an_identical_corpus(self) -> None:
        a = build_labelled_phi_corpus(n_documents=4, seed=42)
        b = build_labelled_phi_corpus(n_documents=4, seed=42)
        assert a == b

    def test_different_seeds_produce_different_values(self) -> None:
        a = build_labelled_phi_corpus(n_documents=4, seed=1)
        b = build_labelled_phi_corpus(n_documents=4, seed=2)
        assert a.documents[0].text != b.documents[0].text

    def test_default_seed_is_stable(self) -> None:
        """Regression guard: the checked-in accuracy baseline depends on
        DEFAULT_SEED never silently changing."""
        assert DEFAULT_SEED == 20260914


class TestAllIdentifiers:
    def test_all_identifiers_matches_every_planted_span_value(self) -> None:
        corpus = build_labelled_phi_corpus(n_documents=3)
        expected = [span.value for doc in corpus.documents for span in doc.spans]
        assert corpus.all_identifiers == expected

    def test_all_identifiers_are_actually_findable_in_their_own_document(self) -> None:
        corpus = build_labelled_phi_corpus(n_documents=3)
        for doc in corpus.documents:
            for span in doc.spans:
                assert span.value in doc.text


def test_the_shared_pytest_fixture_is_wired_up_correctly(
    labelled_phi_corpus: LabelledPHICorpus,
) -> None:
    """Smoke test for tests/conftest.py's session-scoped `labelled_phi_corpus`
    fixture -- the thing every future security/accuracy test will actually
    request, rather than calling build_labelled_phi_corpus() directly."""
    assert labelled_phi_corpus.documents
    assert labelled_phi_corpus.all_identifiers
