"""Unit and property tests for openbtk.deid.consistency.ConsistencyStore.

docs/07_TEST_CHARTER.md section 3.4 names this as one of the properties
stronger than any example: "same (patient, type, value) -> same surrogate,
always." TestSurrogateConsistencyProperty is that property test, run via
Hypothesis rather than a handful of examples.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hypothesis import given
from hypothesis import strategies as st

from openbtk.deid.consistency import ConsistencyStore
from openbtk.deid.schemas import PHICategory

if TYPE_CHECKING:
    from collections.abc import Callable

_KEY = b"0" * 32  # fixed test key -- irrelevant which bytes, just deterministic


def _counting_factory() -> Callable[[], str]:
    """Returns a fresh, distinguishable value each call, so a test can tell
    whether the factory was invoked again or the store returned a cached
    surrogate without calling it."""
    counter = {"n": 0}

    def factory() -> str:
        counter["n"] += 1
        return f"SURROGATE_{counter['n']}"

    return factory


class TestBasicBehaviour:
    def test_first_call_uses_the_factory(self) -> None:
        store = ConsistencyStore(key=_KEY)
        result = store.get_or_create_surrogate(
            patient_id="p1",
            category=PHICategory.NAME,
            original="Jane Doe",
            surrogate_factory=lambda: "PATIENT_1",
        )
        assert result == "PATIENT_1"

    def test_repeated_identical_triple_does_not_call_the_factory_again(self) -> None:
        store = ConsistencyStore(key=_KEY)
        factory = _counting_factory()
        first = store.get_or_create_surrogate(
            patient_id="p1",
            category=PHICategory.NAME,
            original="Jane Doe",
            surrogate_factory=factory,
        )
        second = store.get_or_create_surrogate(
            patient_id="p1",
            category=PHICategory.NAME,
            original="Jane Doe",
            surrogate_factory=factory,
        )
        assert first == second == "SURROGATE_1"

    def test_different_original_gets_a_different_surrogate(self) -> None:
        store = ConsistencyStore(key=_KEY)
        factory = _counting_factory()
        a = store.get_or_create_surrogate(
            patient_id="p1",
            category=PHICategory.NAME,
            original="Jane Doe",
            surrogate_factory=factory,
        )
        b = store.get_or_create_surrogate(
            patient_id="p1",
            category=PHICategory.NAME,
            original="John Smith",
            surrogate_factory=factory,
        )
        assert a != b

    def test_same_value_different_patient_gets_a_different_surrogate(self) -> None:
        """A name shared by two different patients must not collide --
        consistency is per-patient, not global."""
        store = ConsistencyStore(key=_KEY)
        factory = _counting_factory()
        a = store.get_or_create_surrogate(
            patient_id="p1",
            category=PHICategory.NAME,
            original="Jane Doe",
            surrogate_factory=factory,
        )
        b = store.get_or_create_surrogate(
            patient_id="p2",
            category=PHICategory.NAME,
            original="Jane Doe",
            surrogate_factory=factory,
        )
        assert a != b

    def test_same_value_different_category_gets_a_different_surrogate(self) -> None:
        """The same string could plausibly be an MRN in one field and an
        account number in another -- must not collide."""
        store = ConsistencyStore(key=_KEY)
        factory = _counting_factory()
        a = store.get_or_create_surrogate(
            patient_id="p1",
            category=PHICategory.MEDICAL_RECORD_NUMBER,
            original="123456",
            surrogate_factory=factory,
        )
        b = store.get_or_create_surrogate(
            patient_id="p1",
            category=PHICategory.ACCOUNT_NUMBER,
            original="123456",
            surrogate_factory=factory,
        )
        assert a != b

    def test_len_counts_distinct_triples(self) -> None:
        store = ConsistencyStore(key=_KEY)
        store.get_or_create_surrogate(
            patient_id="p1",
            category=PHICategory.NAME,
            original="Jane Doe",
            surrogate_factory=lambda: "A",
        )
        store.get_or_create_surrogate(
            patient_id="p1",
            category=PHICategory.NAME,
            original="Jane Doe",
            surrogate_factory=lambda: "A",
        )
        assert len(store) == 1


class TestNeverStoresTheOriginal:
    def test_state_contains_neither_the_original_nor_the_patient_id(self) -> None:
        store = ConsistencyStore(key=_KEY)
        store.get_or_create_surrogate(
            patient_id="patient-jane-doe-123",
            category=PHICategory.SSN,
            original="123-45-6789",  # phi-fixture-ok
            surrogate_factory=lambda: "[SSN_1]",
        )
        blob = repr(store.state)
        assert "123-45-6789" not in blob  # phi-fixture-ok
        assert "patient-jane-doe-123" not in blob

    def test_state_values_are_only_the_surrogates_supplied(self) -> None:
        store = ConsistencyStore(key=_KEY)
        store.get_or_create_surrogate(
            patient_id="p1",
            category=PHICategory.NAME,
            original="Jane Doe",
            surrogate_factory=lambda: "PATIENT_1",
        )
        assert list(store.state.values()) == ["PATIENT_1"]


class TestKeyedCrossRunConsistency:
    def test_two_stores_with_the_same_key_produce_the_same_digest(self) -> None:
        """The whole point of a keyed HMAC: a second ConsistencyStore,
        constructed later (a later run) with the same key, must compute the
        identical digest for the identical triple -- required for
        initial_state loaded from a persisted run to actually match up."""
        store_a = ConsistencyStore(key=_KEY)
        store_a.get_or_create_surrogate(
            patient_id="p1",
            category=PHICategory.NAME,
            original="Jane Doe",
            surrogate_factory=lambda: "PATIENT_1",
        )
        store_b = ConsistencyStore(key=_KEY, initial_state=store_a.state)
        reused = store_b.get_or_create_surrogate(
            patient_id="p1",
            category=PHICategory.NAME,
            original="Jane Doe",
            surrogate_factory=lambda: "SHOULD_NOT_BE_CALLED",
        )
        assert reused == "PATIENT_1"

    def test_two_stores_with_different_keys_do_not_share_state(self) -> None:
        store_a = ConsistencyStore(key=_KEY)
        store_a.get_or_create_surrogate(
            patient_id="p1",
            category=PHICategory.NAME,
            original="Jane Doe",
            surrogate_factory=lambda: "PATIENT_1",
        )
        store_b = ConsistencyStore(key=b"1" * 32, initial_state=store_a.state)
        fresh = store_b.get_or_create_surrogate(
            patient_id="p1",
            category=PHICategory.NAME,
            original="Jane Doe",
            surrogate_factory=lambda: "PATIENT_NEW",
        )
        assert fresh == "PATIENT_NEW"

    def test_no_key_given_generates_a_usably_random_one(self) -> None:
        """Not a security proof -- just confirms two independently
        default-constructed stores don't accidentally share a hardcoded
        key: copying one's state into a fresh, differently-keyed store must
        NOT be recognised, so its factory runs again rather than reusing
        the copied surrogate."""
        a = ConsistencyStore()
        a.get_or_create_surrogate(
            patient_id="p1",
            category=PHICategory.NAME,
            original="Jane Doe",
            surrogate_factory=lambda: "A",
        )
        b = ConsistencyStore(initial_state=a.state)
        result = b.get_or_create_surrogate(
            patient_id="p1",
            category=PHICategory.NAME,
            original="Jane Doe",
            surrogate_factory=lambda: "B_FRESH",
        )
        assert result == "B_FRESH"


class TestSurrogateConsistencyProperty:
    """docs/07_TEST_CHARTER.md section 3.4: same (patient, type, value) ->
    same surrogate, always -- checked via Hypothesis rather than examples."""

    @given(
        patient_id=st.text(min_size=1, max_size=20),
        category=st.sampled_from(list(PHICategory)),
        original=st.text(min_size=1, max_size=50),
        n_repeats=st.integers(min_value=2, max_value=5),
    )
    def test_repeated_lookups_always_return_the_first_surrogate(
        self, patient_id: str, category: PHICategory, original: str, n_repeats: int
    ) -> None:
        store = ConsistencyStore(key=_KEY)
        factory = _counting_factory()
        results = [
            store.get_or_create_surrogate(
                patient_id=patient_id,
                category=category,
                original=original,
                surrogate_factory=factory,
            )
            for _ in range(n_repeats)
        ]
        assert len(set(results)) == 1
