"""``DosePlausibilityGuardrail`` (FR-G-05).

Every limit here belongs to a made-up drug ("examplamine") and is a test value, not
clinical information: OpenBTK ships no dose limits, and these tests must not read
like a formulary.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from openbtk.core.registry import GUARDRAIL_REGISTRY
from openbtk.core.schemas import GuardrailResult, GuardrailSeverity
from openbtk.guardrails.dose import DoseLimit, DosePlausibilityGuardrail

if TYPE_CHECKING:
    from pathlib import Path

_SRC = "synthetic test values, not clinical guidance"


def _limit(**overrides: Any) -> DoseLimit:
    base: dict[str, Any] = {
        "drug": "examplamine",
        "unit": "mg",
        "min_single": 5,
        "max_single": 100,
        "max_daily": 300,
        "routes": ["oral", "intravenous"],
        "source": _SRC,
    }
    return DoseLimit(**{**base, **overrides})


def _guardrail(*limits: DoseLimit) -> DosePlausibilityGuardrail:
    return DosePlausibilityGuardrail(limits=list(limits) or [_limit()])


def _details(result: GuardrailResult) -> dict[str, Any]:
    return dict(result.details)


def _problems(text: str, **kwargs: Any) -> list[dict[str, Any]]:
    guardrail = kwargs.pop("guardrail", None) or _guardrail()
    problems: list[dict[str, Any]] = _details(guardrail.check(text)).get("problems", [])
    return problems


class TestRegistration:
    def test_is_registered_under_its_key(self) -> None:
        assert "guardrail.general.dose_plausibility" in GUARDRAIL_REGISTRY.list_keys()

    def test_constructs_with_no_arguments_and_does_no_io(self) -> None:
        DosePlausibilityGuardrail(reference="/does/not/exist.json")  # no raise


class TestSingleDose:
    def test_over_the_maximum_blocks(self) -> None:
        result = _guardrail().check("Give examplamine 500 mg PO once.")
        assert result.passed is False
        assert result.severity is GuardrailSeverity.BLOCK
        (problem,) = _details(result)["problems"]
        assert problem["problem"] == "over_max_single"
        assert problem["value"] == 500 and problem["limit"] == 100
        assert problem["source"] == _SRC

    def test_within_the_limits_passes(self) -> None:
        result = _guardrail().check("Give examplamine 50 mg PO once.")
        assert result.passed is True
        assert "1 dose statement" in result.message

    def test_at_the_maximum_exactly_passes(self) -> None:
        assert _guardrail().check("examplamine 100 mg").passed is True

    def test_below_the_minimum_is_a_warning_not_a_block(self) -> None:
        result = _guardrail().check("examplamine 1 mg")
        assert result.passed is False and result.severity is GuardrailSeverity.WARNING

    def test_a_range_is_judged_at_both_ends(self) -> None:
        assert _problems("examplamine 50-150 mg")[0]["problem"] == "over_max_single"
        assert _problems("examplamine 1-50 mg")[0]["problem"] == "under_min_single"
        assert _problems("examplamine 10 to 50 mg") == []

    def test_the_dose_may_come_before_the_drug(self) -> None:
        assert _problems("Give 500 mg of examplamine")[0]["problem"] == (
            "over_max_single"
        )

    def test_matching_ignores_case_and_needs_whole_words(self) -> None:
        assert _problems("EXAMPLAMINE 500 MG")[0]["problem"] == "over_max_single"
        assert _guardrail().check("nonexamplamine 500 mg").passed is True
        assert _guardrail().check("examplamines 500 mg").passed is True

    def test_an_alias_is_matched(self) -> None:
        g = _guardrail(_limit(aliases=["exmpl"]))
        assert _problems("exmpl 500 mg", guardrail=g)[0]["problem"] == "over_max_single"


class TestUnits:
    def test_mass_units_are_converted(self) -> None:
        assert _problems("examplamine 0.5 g")[0]["problem"] == "over_max_single"
        assert _problems("examplamine 0.05 g") == []  # 50 mg
        assert _problems("examplamine 50000 mcg") == []  # 50 mg
        assert _problems("examplamine 500000 mcg")[0]["problem"] == "over_max_single"

    def test_microgram_spellings_are_understood(self) -> None:
        assert _problems("examplamine 50000 µg") == []
        assert _problems("examplamine 50000 μg") == []

    def test_a_unit_that_cannot_be_compared_is_flagged_not_guessed(self) -> None:
        problems = _problems("examplamine 5 mL")
        assert problems[0]["problem"] == "unit_mismatch"
        assert problems[0]["stated_unit"] == "mL"

    def test_a_non_mass_reference_unit_is_compared_in_that_unit(self) -> None:
        g = _guardrail(_limit(drug="examplin", unit="units", max_single=20))
        assert _problems("examplin 50 units", guardrail=g)[0]["problem"] == (
            "over_max_single"
        )
        assert _problems("examplin 10 units", guardrail=g) == []


class TestDailyTotal:
    @pytest.mark.parametrize(
        ("frequency", "per_day"),
        [
            ("BID", 2),
            ("twice daily", 2),
            ("TID", 3),
            ("three times a day", 3),
            ("QID", 4),
            ("q8h", 3),
            ("every 6 hours", 4),
            ("every 12 hours", 2),
        ],
    )
    def test_the_daily_total_is_computed_from_the_frequency(
        self, frequency: str, per_day: int
    ) -> None:
        # 90 mg is fine alone, so any block is the daily total, 90 x per_day > 300 only
        # when per_day >= 4.
        problems = _problems(f"examplamine 90 mg PO {frequency}")
        over = [p for p in problems if p["problem"] == "over_max_daily"]
        assert bool(over) is (90 * per_day > 300)
        if over:
            assert over[0]["doses_per_day"] == per_day
            assert over[0]["value"] == 90 * per_day

    def test_no_frequency_means_the_daily_limit_is_not_checked(self) -> None:
        assert _problems("examplamine 90 mg PO") == []

    def test_an_unreadable_frequency_is_not_guessed(self) -> None:
        assert _problems("examplamine 90 mg PO q0h") == []

    def test_an_over_daily_total_blocks(self) -> None:
        result = _guardrail().check("examplamine 100 mg PO QID")
        assert result.severity is GuardrailSeverity.BLOCK


class TestRoute:
    def test_a_listed_route_passes(self) -> None:
        assert _problems("examplamine 50 mg IV") == []

    def test_an_unlisted_route_is_a_warning(self) -> None:
        result = _guardrail().check("examplamine 50 mg IM")
        assert result.severity is GuardrailSeverity.WARNING
        assert _details(result)["problems"][0]["route"] == "intramuscular"

    def test_no_route_in_the_text_is_not_a_problem(self) -> None:
        assert _problems("examplamine 50 mg") == []

    def test_no_routes_in_the_reference_means_no_route_check(self) -> None:
        g = _guardrail(_limit(routes=[]))
        assert _problems("examplamine 50 mg IM", guardrail=g) == []


class TestClauses:
    def test_two_drugs_are_not_judged_against_each_others_dose(self) -> None:
        g = _guardrail(_limit(), _limit(drug="otherol", max_single=10, max_daily=None))
        # otherol 5 mg is fine; examplamine 500 mg is over. Neither borrows the other's.
        problems = _problems("otherol and examplamine 500 mg", guardrail=g)
        assert [p["drug"] for p in problems] == ["examplamine"]

    def test_a_dose_in_another_sentence_is_not_attributed(self) -> None:
        assert _guardrail().check("Consider examplamine. Weight 500 mg/kg.").passed

    def test_each_mention_is_checked(self) -> None:
        text = "examplamine 50 mg PO. Later, examplamine 500 mg PO."
        assert [p["problem"] for p in _problems(text)] == ["over_max_single"]


class TestNothingCheckedIsSaidPlainly:
    def test_a_drug_outside_the_reference_is_not_checked_and_says_so(self) -> None:
        result = _guardrail().check("Give unrelatedol 9000 mg.")
        assert result.passed is True
        assert "not checked" in result.message

    def test_no_reference_is_a_warning_never_a_silent_pass(self) -> None:
        result = DosePlausibilityGuardrail().check("examplamine 500 mg")
        assert result.passed is False
        assert result.severity is GuardrailSeverity.WARNING
        assert "no dose was checked" in result.message

    def test_an_unreadable_reference_file_is_a_warning(self, tmp_path: Path) -> None:
        g = DosePlausibilityGuardrail(reference=str(tmp_path / "missing.json"))
        result = g.check("examplamine 500 mg")
        assert result.passed is False
        assert result.severity is GuardrailSeverity.WARNING

    def test_a_non_text_payload_is_not_applicable(self) -> None:
        assert _guardrail().check({"a": 1}).passed is True


class TestReferenceFile:
    def _write(self, tmp_path: Path, rows: Any) -> str:
        path = tmp_path / "limits.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        return str(path)

    def test_limits_are_read_from_a_json_file(self, tmp_path: Path) -> None:
        rows = [_limit().model_dump()]
        g = DosePlausibilityGuardrail(reference=self._write(tmp_path, rows))
        assert g.check("examplamine 500 mg").severity is GuardrailSeverity.BLOCK

    def test_a_file_and_inline_limits_combine(self, tmp_path: Path) -> None:
        rows = [_limit().model_dump()]
        g = DosePlausibilityGuardrail(
            reference=self._write(tmp_path, rows),
            limits=[{"drug": "otherol", "max_single": 1, "source": _SRC}],
        )
        assert g.check("otherol 5 mg").severity is GuardrailSeverity.BLOCK
        assert g.check("examplamine 500 mg").severity is GuardrailSeverity.BLOCK

    def test_a_limit_without_a_source_is_refused(self, tmp_path: Path) -> None:
        rows = [{"drug": "examplamine", "max_single": 1}]
        g = DosePlausibilityGuardrail(reference=self._write(tmp_path, rows))
        assert g.check("examplamine 5 mg").passed is False  # WARNING, not a crash

    def test_a_file_that_is_not_a_list_is_refused(self, tmp_path: Path) -> None:
        g = DosePlausibilityGuardrail(reference=self._write(tmp_path, {"a": 1}))
        assert g.check("examplamine 5 mg").severity is GuardrailSeverity.WARNING


class TestDoseLimitValidation:
    def test_source_is_required(self) -> None:
        with pytest.raises(ValidationError):
            DoseLimit(drug="x", max_single=1)  # type: ignore[call-arg]

    def test_min_may_not_exceed_max(self) -> None:
        with pytest.raises(ValidationError, match="min_single"):
            _limit(min_single=10, max_single=5)

    def test_an_unknown_route_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="unknown route"):
            _limit(routes=["through-the-ear"])

    def test_unknown_fields_are_refused(self) -> None:
        with pytest.raises(ValidationError):
            DoseLimit(drug="x", source="s", surprise=1)  # type: ignore[call-arg]


class TestProvenanceAndContent:
    def test_provenance_records_a_hash_and_a_count_not_the_table(self) -> None:
        cfg = _guardrail().provenance().config
        assert cfg["limits"] == 1
        assert (
            isinstance(cfg["reference_sha256"], str)
            and len(cfg["reference_sha256"]) == 64
        )
        assert "examplamine" not in json.dumps(cfg)

    def test_the_hash_changes_when_a_limit_changes(self) -> None:
        a = _guardrail(_limit(max_single=100)).provenance().config["reference_sha256"]
        b = _guardrail(_limit(max_single=101)).provenance().config["reference_sha256"]
        assert a != b

    def test_the_hash_does_not_depend_on_order(self) -> None:
        one, two = _limit(), _limit(drug="otherol")
        a = _guardrail(one, two).provenance().config["reference_sha256"]
        b = _guardrail(two, one).provenance().config["reference_sha256"]
        assert a == b

    def test_an_unreadable_reference_does_not_break_provenance(self) -> None:
        g = DosePlausibilityGuardrail(reference="/does/not/exist.json")
        assert g.provenance().config == {"reference": "unreadable"}

    def test_a_result_never_carries_the_surrounding_text(self) -> None:
        text = "Patient Zebediah Quux is prescribed examplamine 500 mg PO today."
        dumped = _guardrail().check(text).model_dump_json()
        assert "Zebediah" not in dumped and "Quux" not in dumped
