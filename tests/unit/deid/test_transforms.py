"""Unit and property tests for openbtk.deid.transforms.Transform.

docs/07_TEST_CHARTER.md section 3.4 names the property this file's
TestDateShiftIntervalProperty checks directly: "date_shift: date2 - date1
preserved for any pair within a patient."
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st

from openbtk.core.errors import DeidError
from openbtk.core.schemas import TextSpan
from openbtk.deid.consistency import ConsistencyStore
from openbtk.deid.schemas import DeidMode, Detection, PHICategory
from openbtk.deid.transforms import Transform

_KEY = b"0" * 32


def _det(category: PHICategory, start: int, end: int) -> Detection:
    return Detection(
        category=category,
        span=TextSpan(start=start, end=end, label=category.value, confidence=0.9),
        confidence=0.9,
        method="ensemble",
    )


class TestRedact:
    def test_replaces_the_span_with_a_uniform_tag(self) -> None:
        text = "SSN: 123-45-6789."  # phi-fixture-ok
        det = _det(PHICategory.SSN, 5, 16)
        result = Transform(DeidMode.REDACT).apply(text, [det])
        assert result == "SSN: [REDACTED]."

    def test_reveals_no_category_information(self) -> None:
        text = "Name: Jane Doe."
        det = _det(PHICategory.NAME, 6, 14)
        result = Transform(DeidMode.REDACT).apply(text, [det])
        assert "NAME" not in result
        assert "Jane Doe" not in result


class TestTag:
    def test_replaces_the_span_with_its_category_tag(self) -> None:
        text = "Name: Jane Doe."
        det = _det(PHICategory.NAME, 6, 14)
        result = Transform(DeidMode.TAG).apply(text, [det])
        assert result == "Name: [NAME]."


class TestHash:
    def test_replaces_with_a_deterministic_keyed_hash_token(self) -> None:
        text = "SSN: 123-45-6789."  # phi-fixture-ok
        det = _det(PHICategory.SSN, 5, 16)
        transform = Transform(DeidMode.HASH, key=_KEY)
        result = transform.apply(text, [det])
        assert result.startswith("SSN: [SSN_HASH_")
        assert "123-45-6789" not in result  # phi-fixture-ok

    def test_same_key_and_value_produce_the_same_token_across_instances(self) -> None:
        text = "SSN: 123-45-6789."  # phi-fixture-ok
        det = _det(PHICategory.SSN, 5, 16)
        a = Transform(DeidMode.HASH, key=_KEY).apply(text, [det])
        b = Transform(DeidMode.HASH, key=_KEY).apply(text, [det])
        assert a == b

    def test_different_keys_produce_different_tokens(self) -> None:
        text = "SSN: 123-45-6789."  # phi-fixture-ok
        det = _det(PHICategory.SSN, 5, 16)
        a = Transform(DeidMode.HASH, key=_KEY).apply(text, [det])
        b = Transform(DeidMode.HASH, key=b"1" * 32).apply(text, [det])
        assert a != b

    def test_no_key_given_still_produces_a_usable_result(self) -> None:
        """Not a security proof -- confirms the random-default path (no
        hardcoded fallback key) actually works end to end."""
        text = "SSN: 123-45-6789."  # phi-fixture-ok
        det = _det(PHICategory.SSN, 5, 16)
        result = Transform(DeidMode.HASH).apply(text, [det])
        assert "123-45-6789" not in result  # phi-fixture-ok


class TestSurrogate:
    def test_first_occurrence_gets_a_numbered_surrogate(self) -> None:
        text = "Name: Jane Doe."
        det = _det(PHICategory.NAME, 6, 14)
        result = Transform(DeidMode.SURROGATE).apply(text, [det], patient_id="p1")
        assert result == "Name: [NAME_1]."

    def test_repeated_value_for_the_same_patient_reuses_the_surrogate(self) -> None:
        transform = Transform(DeidMode.SURROGATE)
        text1 = "Name: Jane Doe."
        det1 = _det(PHICategory.NAME, 6, 14)
        text2 = "Re: Jane Doe again."
        det2 = _det(PHICategory.NAME, 4, 12)
        first = transform.apply(text1, [det1], patient_id="p1")
        second = transform.apply(text2, [det2], patient_id="p1")
        assert "[NAME_1]" in first
        assert "[NAME_1]" in second

    def test_different_values_get_different_numbered_surrogates(self) -> None:
        transform = Transform(DeidMode.SURROGATE)
        text = "Jane Doe and John Smith."
        dets = [_det(PHICategory.NAME, 0, 8), _det(PHICategory.NAME, 13, 23)]
        result = transform.apply(text, dets, patient_id="p1")
        assert result == "[NAME_1] and [NAME_2]."

    def test_different_patients_do_not_share_surrogate_numbering_state(self) -> None:
        """The ConsistencyStore keys on patient_id too, but the sequential
        counter is per-Transform-instance, per-category -- confirms a
        second patient's first name doesn't collide with or continue the
        first patient's numbering by accident."""
        transform = Transform(DeidMode.SURROGATE)
        text = "Jane Doe"
        det = _det(PHICategory.NAME, 0, 8)
        first_patient = transform.apply(text, [det], patient_id="p1")
        second_patient = transform.apply(text, [det], patient_id="p2")
        assert first_patient == "[NAME_1]"
        assert second_patient == "[NAME_2]"  # a NEW triple -> a new counter tick

    def test_can_share_a_consistency_store_for_cross_run_persistence(self) -> None:
        store = ConsistencyStore(key=_KEY)
        text = "Jane Doe"
        det = _det(PHICategory.NAME, 0, 8)
        first_run = Transform(DeidMode.SURROGATE, consistency_store=store)
        result_a = first_run.apply(text, [det], patient_id="p1")

        second_run = Transform(
            DeidMode.SURROGATE,
            consistency_store=ConsistencyStore(key=_KEY, initial_state=store.state),
        )
        result_b = second_run.apply(text, [det], patient_id="p1")
        assert result_a == result_b


class TestDateShift:
    def test_shifts_a_slash_formatted_date(self) -> None:
        text = "DOB: 03/14/1958."
        det = _det(PHICategory.DATE, 5, 15)
        result = Transform(DeidMode.DATE_SHIFT, key=_KEY).apply(
            text, [det], patient_id="p1"
        )
        assert "03/14/1958" not in result
        # Output is always ISO 8601, regardless of input format.
        shifted_str = result.removeprefix("DOB: ").removesuffix(".")
        date.fromisoformat(shifted_str)  # raises if not a valid ISO date

    def test_shifts_an_iso_formatted_date(self) -> None:
        text = "DOB: 1958-03-14."
        det = _det(PHICategory.DATE, 5, 15)
        result = Transform(DeidMode.DATE_SHIFT, key=_KEY).apply(
            text, [det], patient_id="p1"
        )
        assert "1958-03-14" not in result

    def test_unparseable_date_raises_deid_error(self) -> None:
        text = "DOB: not-a-date."
        det = _det(PHICategory.DATE, 5, 15)
        with pytest.raises(DeidError):
            Transform(DeidMode.DATE_SHIFT, key=_KEY).apply(text, [det], patient_id="p1")

    def test_same_patient_gets_the_same_shift_amount_every_time(self) -> None:
        transform = Transform(DeidMode.DATE_SHIFT, key=_KEY)
        det1 = _det(PHICategory.DATE, 0, 10)
        det2 = _det(PHICategory.DATE, 0, 10)
        a = transform.apply("01/01/2000", [det1], patient_id="p1")
        b = transform.apply("06/15/2005", [det2], patient_id="p1")
        shift_a = (date.fromisoformat(a) - date(2000, 1, 1)).days
        shift_b = (date.fromisoformat(b) - date(2005, 6, 15)).days
        assert shift_a == shift_b

    def test_different_patients_get_different_shift_amounts(self) -> None:
        """Not guaranteed mathematically (two independent HMAC outputs
        could coincidentally collide), but true for this fixed key and
        these two patient ids -- a real regression guard, not a proof."""
        transform = Transform(DeidMode.DATE_SHIFT, key=_KEY)
        a = transform.apply(
            "01/01/2000", [_det(PHICategory.DATE, 0, 10)], patient_id="p1"
        )
        b = transform.apply(
            "01/01/2000", [_det(PHICategory.DATE, 0, 10)], patient_id="p2"
        )
        assert a != b


class TestOverlapDefence:
    def test_overlapping_detections_raise_deid_error(self) -> None:
        text = "0123456789"
        overlapping = [
            _det(PHICategory.SSN, 0, 5),
            _det(PHICategory.OTHER_UNIQUE_IDENTIFIER, 3, 8),
        ]
        with pytest.raises(DeidError):
            Transform(DeidMode.REDACT).apply(text, overlapping)


class TestNoDetections:
    def test_text_with_no_detections_passes_through_unchanged(self) -> None:
        text = "Nothing sensitive here."
        assert Transform(DeidMode.REDACT).apply(text, []) == text


class TestDateShiftIntervalProperty:
    """docs/07_TEST_CHARTER.md section 3.4: date_shift preserves date2 -
    date1 for any pair within a patient -- checked via Hypothesis."""

    @given(
        patient_id=st.text(min_size=1, max_size=20),
        base=st.dates(min_value=date(1900, 1, 1), max_value=date(2100, 1, 1)),
        offset_days=st.integers(min_value=-3650, max_value=3650),
    )
    def test_interval_between_two_dates_is_preserved_after_shifting(
        self, patient_id: str, base: date, offset_days: int
    ) -> None:
        other = base + timedelta(days=offset_days)
        transform = Transform(DeidMode.DATE_SHIFT, key=_KEY)
        shifted_base = date.fromisoformat(
            transform.apply(
                base.isoformat(), [_det(PHICategory.DATE, 0, 10)], patient_id=patient_id
            )
        )
        shifted_other = date.fromisoformat(
            transform.apply(
                other.isoformat(),
                [_det(PHICategory.DATE, 0, 10)],
                patient_id=patient_id,
            )
        )
        assert (shifted_other - shifted_base).days == offset_days
