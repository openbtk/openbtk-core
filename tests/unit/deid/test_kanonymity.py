"""k-anonymity measurement and generalisation (FR-D-11).

All data here is synthetic and made up in the tests.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from openbtk.deid.kanonymity import (
    DEFAULT_LADDERS,
    anonymise_to_k,
    band,
    k_anonymity_report,
    prefix,
    star,
    top_code,
    zip_ladder,
)


def _rows(spec: list[tuple[dict[str, Any], int]]) -> list[dict[str, Any]]:
    return [dict(row) for row, count in spec for _ in range(count)]


class TestReport:
    def test_counts_classes_and_the_smallest(self) -> None:
        rows = _rows([({"sex": "f", "yob": 1980}, 4), ({"sex": "m", "yob": 1975}, 1)])
        report = k_anonymity_report(rows, ["sex", "yob"], k=3)
        assert report.n_records == 5 and report.n_classes == 2
        assert report.k_achieved == 1 and report.n_unique == 1
        assert report.n_below_k == 1 and report.satisfied is False
        assert report.smallest_classes == [1, 4]

    def test_a_satisfied_table(self) -> None:
        rows = _rows([({"sex": "f"}, 3), ({"sex": "m"}, 3)])
        report = k_anonymity_report(rows, ["sex"], k=3)
        assert report.satisfied and report.k_achieved == 3 and report.n_below_k == 0

    def test_an_empty_table_is_vacuously_satisfied(self) -> None:
        report = k_anonymity_report([], ["sex"], k=5)
        assert report.satisfied and report.n_records == 0 and report.k_achieved == 0

    def test_a_missing_value_is_its_own_class(self) -> None:
        rows = [{"sex": None}, {}, {"sex": "f"}]
        report = k_anonymity_report(rows, ["sex"], k=2)
        assert report.n_classes == 2  # None and absent are the same "unknown"
        assert report.k_achieved == 1

    def test_only_the_named_columns_count(self) -> None:
        rows = [{"sex": "f", "note": "a"}, {"sex": "f", "note": "b"}]
        assert k_anonymity_report(rows, ["sex"], k=2).satisfied

    def test_it_streams_its_input(self) -> None:
        consumed = 0

        def rows() -> Any:
            nonlocal consumed
            for _ in range(1000):
                consumed += 1
                yield {"sex": "f"}

        report = k_anonymity_report(rows(), ["sex"], k=5)
        assert report.n_records == 1000 and consumed == 1000

    def test_the_report_holds_counts_never_values(self) -> None:
        rows = _rows([({"zip": "90210-SENTINEL"}, 1), ({"zip": "10001"}, 3)])
        dumped = k_anonymity_report(rows, ["zip"], k=2).model_dump_json()
        assert "SENTINEL" not in dumped and "90210" not in dumped

    @pytest.mark.parametrize("bad_k", [0, -1])
    def test_k_must_be_positive(self, bad_k: int) -> None:
        with pytest.raises(ValueError, match="k must"):
            k_anonymity_report([], ["sex"], k=bad_k)

    def test_a_quasi_identifier_is_required(self) -> None:
        with pytest.raises(ValueError, match="quasi-identifier"):
            k_anonymity_report([], [], k=2)


class TestLadders:
    def test_band(self) -> None:
        assert band(5)(1987) == "1985-1989"
        assert band(10)(2000) == "2000-2009"
        assert band(5)(None) is None  # unknown stays unknown
        assert band(5)("x") == "x"

    def test_band_top_codes(self) -> None:
        assert band(10, top=90)(93) == "90+"
        assert band(10, top=90)(89) == "80-89"

    def test_bool_is_not_a_number(self) -> None:
        assert band(5)(True) is True

    def test_band_width_must_be_at_least_two(self) -> None:
        with pytest.raises(ValueError, match="width"):
            band(1)

    def test_top_code(self) -> None:
        assert top_code(90)(90) == "90+" and top_code(90)(89) == 89
        assert top_code(90)(None) is None

    def test_prefix_and_zip_ladder(self) -> None:
        assert prefix(3)("90210") == "902**"
        assert prefix(3)("90") == "90"
        assert [f("90210") for f in zip_ladder(5)] == [
            "90210",
            "9021*",
            "902**",
            "90***",
            "9****",
            "*",
        ]

    def test_prefix_needs_a_digit_count(self) -> None:
        with pytest.raises(ValueError, match="digits"):
            prefix(0)

    def test_star_hides_everything(self) -> None:
        assert star(1) == star("x") == star(None) == "*"

    def test_age_is_top_coded_at_ninety_from_the_first_level(self) -> None:
        """HIPAA Safe Harbor: ages over 89 are aggregated. Level 0 already does it."""
        level0 = DEFAULT_LADDERS["age"][0]
        assert level0(93) == "90+" and level0(41) == 41

    def test_dates_become_iso_strings(self) -> None:
        from datetime import date

        assert DEFAULT_LADDERS["gender"][0](date(2024, 3, 1)) == "2024-03-01"


class TestAnonymise:
    def _people(self) -> list[dict[str, Any]]:
        return [
            {"pid": f"p{i}", "yob": 1970 + (i % 20), "sex": "f" if i % 2 else "m"}
            for i in range(60)
        ]

    def test_reaches_k_and_reports_it(self) -> None:
        result = anonymise_to_k(self._people(), ["yob", "sex"], k=5, keep=["pid"])
        assert result.report.satisfied and result.report.k_achieved >= 5
        assert result.n_input == 60 and len(result.rows) + result.n_suppressed == 60

    def test_the_output_is_independently_k_anonymous(self) -> None:
        result = anonymise_to_k(self._people(), ["yob", "sex"], k=5)
        assert k_anonymity_report(result.rows, ["yob", "sex"], k=5).satisfied

    def test_generalisation_is_used_before_suppression(self) -> None:
        result = anonymise_to_k(self._people(), ["yob", "sex"], k=5)
        assert result.n_suppressed == 0
        assert result.levels["yob"] > 0

    def test_an_already_k_anonymous_table_is_left_alone(self) -> None:
        rows = _rows([({"sex": "f"}, 5), ({"sex": "m"}, 5)])
        result = anonymise_to_k(rows, ["sex"], k=5)
        assert result.levels == {"sex": 0} and result.n_suppressed == 0
        assert {r["sex"] for r in result.rows} == {"f", "m"}

    def test_unnamed_columns_are_dropped_and_keep_columns_pass_through(self) -> None:
        rows = [{"sex": "f", "note": "SECRET", "pid": "a"}] * 5
        result = anonymise_to_k(rows, ["sex"], k=5, keep=["pid"])
        assert set(result.rows[0]) == {"sex", "pid"}
        assert "SECRET" not in result.model_dump_json()

    def test_a_table_smaller_than_k_ends_empty_and_says_so(self) -> None:
        rows = _rows([({"sex": "f"}, 2)])
        result = anonymise_to_k(rows, ["sex"], k=5)
        assert result.rows == [] and result.n_suppressed == 2

    def test_a_few_outliers_are_suppressed_not_generalised_away(self) -> None:
        rows = _rows([({"sex": "f"}, 50), ({"sex": "m"}, 50), ({"sex": "x"}, 1)])
        result = anonymise_to_k(rows, ["sex"], k=5, max_suppression=0.05)
        assert result.n_suppressed == 1 and result.levels == {"sex": 0}

    def test_no_suppression_allowed_generalises_instead(self) -> None:
        rows = _rows([({"sex": "f"}, 50), ({"sex": "m"}, 50), ({"sex": "x"}, 1)])
        result = anonymise_to_k(rows, ["sex"], k=5, max_suppression=0)
        assert result.levels == {"sex": 1} and result.n_suppressed == 0
        assert {r["sex"] for r in result.rows} == {"*"}

    def test_a_custom_ladder_is_used(self) -> None:
        rows = [{"zip": f"9021{i % 3}"} for i in range(9)]
        result = anonymise_to_k(rows, ["zip"], k=9, ladders={"zip": zip_ladder(5)})
        assert result.report.satisfied
        # "9021*" is the first level that puts all nine in one class
        assert {r["zip"] for r in result.rows} == {"9021*"}

    def test_the_attribute_with_most_distinct_values_is_generalised_first(self) -> None:
        rows = [{"a": i, "b": i % 2} for i in range(40)]
        ladders = {"a": [lambda v: v, band(10), star], "b": [lambda v: v, star]}
        result = anonymise_to_k(rows, ["a", "b"], k=5, ladders=ladders)
        assert result.levels["a"] >= 1 and result.levels["b"] == 0

    def test_it_is_deterministic(self) -> None:
        a = anonymise_to_k(self._people(), ["yob", "sex"], k=5)
        b = anonymise_to_k(self._people(), ["yob", "sex"], k=5)
        assert a == b

    def test_results_are_json_serialisable(self) -> None:
        from datetime import date

        rows = [{"d": date(2024, 3, 1), "sex": "f"}] * 5
        result = anonymise_to_k(rows, ["d", "sex"], k=5)
        assert json.loads(result.model_dump_json())["rows"][0]["d"] == "2024-03-01"

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"k": 0}, "k must"),
            ({"k": 2, "max_suppression": 2}, "max_suppression"),
        ],
    )
    def test_bad_arguments_are_refused(
        self, kwargs: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            anonymise_to_k([{"sex": "f"}], ["sex"], **kwargs)

    def test_an_empty_or_over_long_ladder_is_refused(self) -> None:
        with pytest.raises(ValueError, match="ladder"):
            anonymise_to_k([{"a": 1}], ["a"], k=2, ladders={"a": []})
        with pytest.raises(ValueError, match="ladder"):
            anonymise_to_k([{"a": 1}], ["a"], k=2, ladders={"a": [star] * 13})


_VALUE = st.one_of(st.none(), st.integers(1900, 2020), st.sampled_from(["f", "m", "x"]))


@settings(max_examples=60, deadline=None)
@given(
    rows=st.lists(
        st.fixed_dictionaries(
            {"a": _VALUE, "b": st.sampled_from(["f", "m", "x", None])}
        ),
        max_size=80,
    ),
    k=st.integers(1, 6),
    max_suppression=st.sampled_from([0.0, 0.05, 0.5, 1.0]),
)
def test_the_output_is_always_k_anonymous_and_accounts_for_every_record(
    rows: list[dict[str, Any]], k: int, max_suppression: float
) -> None:
    result = anonymise_to_k(rows, ["a", "b"], k=k, max_suppression=max_suppression)
    assert k_anonymity_report(result.rows, ["a", "b"], k=k).satisfied
    assert len(result.rows) + result.n_suppressed == len(rows) == result.n_input
    assert result.report.n_records == len(result.rows)
