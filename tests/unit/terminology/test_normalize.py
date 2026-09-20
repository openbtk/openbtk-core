"""``ConceptNormalizer`` (FR-M-06). Vocabularies here are tiny and synthetic; the LOINC
code ``58410-2`` is the one the PRD names, with a display invented for the test (no
licensed vocabulary text is used)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from openbtk.core.errors import TerminologyError
from openbtk.core.schemas import CodeSystem, Concept
from openbtk.terminology import ConceptNormalizer
from openbtk.terminology.normalize import ConceptMatch

if TYPE_CHECKING:
    from pathlib import Path


def _concept(
    code: str, display: str, system: CodeSystem = CodeSystem.ICD10CM
) -> Concept:
    return Concept(code=code, system=system, display=display)


_VOCAB = [
    _concept("I10", "Essential (primary) hypertension"),
    _concept("E11.9", "Type 2 diabetes mellitus without complications"),
    _concept("E10.9", "Type 1 diabetes mellitus without complications"),
    _concept("M25.561", "Pain in right knee"),
    _concept("M25.562", "Pain in left knee"),
    _concept("N17.9", "Acute kidney failure, unspecified"),
    _concept("N18.9", "Chronic kidney disease, unspecified"),
    _concept("58410-2", "Synthetic full blood count panel", CodeSystem.LOINC),
]
_ALIASES = {"58410-2": ["CBC", "complete blood count"], "I10": ["HTN"]}


def _normalizer(**kwargs: float) -> ConceptNormalizer:
    return ConceptNormalizer(_VOCAB, aliases=_ALIASES, **kwargs)


class TestExactAndAlias:
    def test_the_display_matches_exactly(self) -> None:
        match = _normalizer().resolve("Essential (primary) hypertension")
        assert match is not None
        assert (match.concept.code, match.method, match.score) == ("I10", "exact", 1.0)

    def test_case_punctuation_and_accents_do_not_matter(self) -> None:
        for text in (
            "ESSENTIAL   primary HYPERTENSION!",
            "essential, primary; hypertension",
        ):
            assert _normalizer().resolve(text).concept.code == "I10"  # type: ignore[union-attr]
        accented = ConceptNormalizer([_concept("X1", "Anemia café")])
        assert accented.resolve("anemia cafe").concept.code == "X1"  # type: ignore[union-attr]

    def test_cbc_reaches_its_loinc_code_through_an_alias(self) -> None:
        match = _normalizer().resolve("CBC")
        assert match is not None
        assert (
            match.concept.code == "58410-2" and match.concept.system is CodeSystem.LOINC
        )
        assert (match.method, match.matched_text, match.score) == ("alias", "CBC", 1.0)

    def test_without_the_alias_cbc_is_not_guessed(self) -> None:
        assert ConceptNormalizer(_VOCAB).resolve("CBC") is None

    def test_an_alias_on_two_codes_is_ambiguous(self) -> None:
        normalizer = ConceptNormalizer(
            [_concept("A1", "Alpha"), _concept("B1", "Beta")],
            aliases={"A1": ["shared"], "B1": ["shared"]},
        )
        assert normalizer.resolve("shared") is None
        assert {m.concept.code for m in normalizer.normalise("shared")} == {"A1", "B1"}


class TestFuzzy:
    def test_a_typo_matches_and_is_labelled_fuzzy(self) -> None:
        match = _normalizer().resolve("essential primary hypertenson")
        assert match is not None and match.concept.code == "I10"
        assert match.method == "fuzzy" and 0.85 <= match.score < 1.0

    def test_a_fuzzy_score_is_never_reported_as_one(self) -> None:
        match = _normalizer().resolve("essential primary hypertensionn")
        assert match is not None and match.method == "fuzzy" and match.score < 1.0

    def test_unrelated_text_matches_nothing(self) -> None:
        assert _normalizer().normalise("fractured femur") == []

    def test_a_higher_threshold_narrows_and_one_allows_only_exact(self) -> None:
        assert (
            _normalizer(threshold=1.0).resolve("essential primary hypertenson") is None
        )
        assert _normalizer(threshold=1.0).resolve("htn") is not None

    def test_a_lower_threshold_widens(self) -> None:
        text = "essential hypertension"  # a word short of the display
        assert _normalizer().resolve(text) is None
        assert _normalizer(threshold=0.7).resolve(text).concept.code == "I10"  # type: ignore[union-attr]

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_thresholds_outside_zero_one_are_refused(self, bad: float) -> None:
        with pytest.raises(ValueError, match="threshold"):
            ConceptNormalizer([], threshold=bad)
        with pytest.raises(ValueError, match="ambiguity_margin"):
            ConceptNormalizer([], ambiguity_margin=bad)


class TestMeaningChangingModifiers:
    """Plain string similarity would happily match all of these. They must not."""

    @pytest.mark.parametrize(
        ("text", "wrong_code"),
        [
            ("Type 3 diabetes mellitus without complications", "E11.9"),
            ("Type 2 diabetes mellitus without complications", "E10.9"),
            ("Pain in left knee", "M25.561"),
            ("Pain in right knee", "M25.562"),
            ("Chronic kidney failure, unspecified", "N17.9"),
            ("Acute kidney disease, unspecified", "N18.9"),
        ],
    )
    def test_a_differing_modifier_never_matches(
        self, text: str, wrong_code: str
    ) -> None:
        codes = {
            m.concept.code for m in _normalizer(threshold=0.5).normalise(text, limit=10)
        }
        assert wrong_code not in codes

    def test_the_right_one_is_still_found(self) -> None:
        assert _normalizer().resolve("pain in left knee").concept.code == "M25.562"  # type: ignore[union-attr]
        assert (
            _normalizer().resolve("type 1 diabetes mellitus without complication")
            is not None
        )

    def test_a_negation_word_changes_the_match(self) -> None:
        normalizer = ConceptNormalizer(
            [
                _concept("A", "Diabetes with complications"),
                _concept("B", "Diabetes without complications"),
            ]
        )
        assert normalizer.resolve("diabetes without complication").concept.code == "B"  # type: ignore[union-attr]

    def test_a_sentence_is_not_a_term(self) -> None:
        assert (
            _normalizer().resolve("no history of essential primary hypertension")
            is None
        )


class TestAmbiguity:
    def test_near_ties_between_different_codes_give_no_answer(self) -> None:
        normalizer = ConceptNormalizer(
            [
                _concept("A", "chest pain on exertion"),
                _concept("B", "chest pain on exertions"),
            ]
        )
        assert normalizer.resolve("chest pain on exertio") is None
        assert len(normalizer.normalise("chest pain on exertio")) == 2

    def test_a_wide_margin_refuses_any_two_way_result(self) -> None:
        pair = [
            _concept("A", "chest pain on exertion"),
            _concept("B", "chest pain on exertional"),
        ]
        assert (
            ConceptNormalizer(pair, ambiguity_margin=1.0).resolve(
                "chest pain on exertion"
            )
            is not None
        )  # exact wins
        assert (
            ConceptNormalizer(pair, ambiguity_margin=1.0).resolve(
                "chest pain on exertiona"
            )
            is None
        )

    def test_a_zero_margin_only_refuses_exact_ties(self) -> None:
        loose = ConceptNormalizer(
            [
                _concept("A", "chest pain on exertion"),
                _concept("B", "chest pain on exertions"),
            ],
            ambiguity_margin=0.0,
        )
        assert loose.resolve("chest pain on exertio") is not None


class TestNormalise:
    def test_limit_and_order_are_deterministic(self) -> None:
        vocab = [
            _concept(f"K{i}", f"kidney {word}")
            for i, word in enumerate(["stone", "cyst", "mass", "scar", "stent"])
        ]
        normalizer = ConceptNormalizer(vocab, threshold=0.5)
        first = [m.concept.code for m in normalizer.normalise("kidney ston", limit=5)]
        assert first == [
            m.concept.code for m in normalizer.normalise("kidney ston", limit=5)
        ]
        assert first[0] == "K0"
        assert len(normalizer.normalise("kidney ston", limit=1)) == 1

    def test_scores_are_descending_and_in_range(self) -> None:
        scores = [
            m.score
            for m in ConceptNormalizer(_VOCAB, threshold=0.3).normalise(
                "kidney", limit=10
            )
        ]
        assert scores == sorted(scores, reverse=True)
        assert all(0 <= s <= 1 for s in scores)

    def test_one_match_per_code_even_with_several_aliases(self) -> None:
        matches = _normalizer(threshold=0.4).normalise("blood count", limit=10)
        assert len([m for m in matches if m.concept.code == "58410-2"]) == 1

    def test_a_system_filter_restricts_the_candidates(self) -> None:
        normalizer = ConceptNormalizer(
            [_concept("1", "Anemia"), _concept("2", "Anemia", CodeSystem.SNOMED)]
        )
        only = normalizer.normalise("anemia", system=CodeSystem.SNOMED)
        assert [m.concept.system for m in only] == [CodeSystem.SNOMED]

    @pytest.mark.parametrize("text", ["", "   ", "!!!", "—"])
    def test_empty_or_punctuation_only_terms_match_nothing(self, text: str) -> None:
        assert _normalizer().normalise(text) == []

    def test_an_overlong_term_is_refused_not_scanned(self) -> None:
        assert _normalizer().normalise("hypertension " * 100) == []

    def test_a_zero_limit_returns_nothing(self) -> None:
        assert _normalizer().normalise("htn", limit=0) == []

    def test_a_match_is_a_frozen_pydantic_model(self) -> None:
        match = _normalizer().resolve("htn")
        assert isinstance(match, ConceptMatch)
        assert '"alias"' in match.model_dump_json()


class TestLoading:
    def test_from_csv_with_an_aliases_column(self, tmp_path: Path) -> None:
        path = tmp_path / "vocab.csv"
        path.write_text(
            "code,system,display,aliases\n"
            "58410-2,LOINC,Synthetic full blood count panel,CBC|complete blood count\n"
            "I10,ICD10CM,Essential hypertension,\n",
            encoding="utf-8",
        )
        normalizer = ConceptNormalizer.from_csv(str(path))
        assert normalizer.resolve("cbc").concept.code == "58410-2"  # type: ignore[union-attr]
        assert normalizer.resolve("Essential hypertension").concept.code == "I10"  # type: ignore[union-attr]

    def test_from_csv_without_an_aliases_column(self, tmp_path: Path) -> None:
        path = tmp_path / "vocab.csv"
        path.write_text(
            "code,system,display\nI10,ICD10CM,Essential hypertension\n",
            encoding="utf-8",
        )
        assert ConceptNormalizer.from_csv(str(path)).resolve("essential hypertension")

    def test_a_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(TerminologyError, match="Could not open"):
            ConceptNormalizer.from_csv(str(tmp_path / "nope.csv"))

    def test_a_missing_column(self, tmp_path: Path) -> None:
        path = tmp_path / "v.csv"
        path.write_text("code,display\nI10,Essential hypertension\n", encoding="utf-8")
        with pytest.raises(TerminologyError, match="system"):
            ConceptNormalizer.from_csv(str(path))

    def test_a_bad_row_names_the_line_not_the_value(self, tmp_path: Path) -> None:
        path = tmp_path / "v.csv"
        path.write_text(
            "code,system,display\nI10,NOT-A-SYSTEM,SECRET-VALUE\n", encoding="utf-8"
        )
        with pytest.raises(TerminologyError) as excinfo:
            ConceptNormalizer.from_csv(str(path))
        assert "line 2" in str(excinfo.value) and "SECRET-VALUE" not in str(
            excinfo.value
        )

    def test_the_bundled_icd10cm_subset(self) -> None:
        normalizer = ConceptNormalizer.from_bundled_icd10cm()
        assert normalizer.resolve("essential primary hypertenson").concept.code == "I10"  # type: ignore[union-attr]
        found = normalizer.resolve("type 2 diabetes mellitus without complication")
        assert found is not None and found.concept.code == "E11.9"

    def test_a_shorter_name_is_not_taken_for_the_specific_one(self) -> None:
        """ "Type 2 diabetes mellitus" is not "...without complications": matching them
        would assert that the patient has no complications. The negation word in the
        candidate that the query lacks is exactly what the guard rejects."""
        normalizer = ConceptNormalizer.from_bundled_icd10cm(threshold=0.5)
        assert normalizer.resolve("type 2 diabetes mellitus") is None
        assert (
            normalizer.resolve("type 1 diabetes mellitus without complications") is None
        )


class TestScale:
    def test_a_large_vocabulary_answers_quickly_and_correctly(self) -> None:
        concepts = [
            _concept(f"X{i}", f"synthetic condition number {i} of the body")
            for i in range(20000)
        ]
        normalizer = ConceptNormalizer(concepts)
        started = time.perf_counter()
        for i in range(0, 20000, 400):
            exact = normalizer.resolve(f"synthetic condition number {i} of the body")
            typo = normalizer.resolve(f"synthetic conditon nmber {i} of teh body")
            assert exact is not None and exact.concept.code == f"X{i}"
            # The number is a meaning-changing token, so only X{i} is a candidate.
            assert typo is not None and typo.concept.code == f"X{i}"
        assert time.perf_counter() - started < 3.0

    @pytest.mark.parametrize("text", ["a" * 255, "ab " * 85, "1" * 255, "left " * 51])
    def test_adversarial_terms_are_bounded(self, text: str) -> None:
        normalizer = ConceptNormalizer(_VOCAB * 200, aliases=_ALIASES)
        started = time.perf_counter()
        normalizer.normalise(text)
        assert time.perf_counter() - started < 3.0


@settings(max_examples=60, deadline=None)
@given(
    base=st.sampled_from(
        ["knee pain", "kidney failure", "diabetes mellitus", "heart failure"]
    ),
    modifier=st.sampled_from(
        ["left", "right", "acute", "chronic", "type 1", "type 2", "no"]
    ),
    other=st.sampled_from(
        ["left", "right", "acute", "chronic", "type 1", "type 2", "no"]
    ),
)
def test_a_result_never_disagrees_with_the_query_on_a_meaning_changing_word(
    base: str, modifier: str, other: str
) -> None:
    normalizer = ConceptNormalizer([_concept("C", f"{modifier} {base}")], threshold=0.3)
    for match in normalizer.normalise(f"{other} {base}"):
        if other != modifier:
            raise AssertionError(f"{other!r} matched {modifier!r}")
        assert match.concept.code == "C"
