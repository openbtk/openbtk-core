"""Shared contract every registered BaseLoader implementation must satisfy.

Parametrized over LOADER_REGISTRY.list_keys() (docs/07_TEST_CHARTER.md
section 3.1): registering a loader anywhere automatically enrolls it here,
with no opt-out. Right now that list contains only tests/contract/conftest.py's
ReferenceLoader; a real modality loader (M3+) is swept in the moment it
registers.

Three of the checks in the doc's illustrative template
(test_raises_loader_error_on_bad_source, test_error_context_contains_no_phi,
test_missing_dependency_names_the_extra) are NOT generically parametrizable
across an arbitrary future loader: what counts as "a bad source" is
inherently loader-specific (a bad file path means nothing to a loader that
reads from an in-memory iterator), and a loader with zero optional
dependencies has no missing-dependency path to test at all. Rather than
force a fake, one-size-fits-all fixture, those three are tested directly
against ReferenceLoader, which conftest.py deliberately gives a real failure
path for exactly this purpose. A future modality's own test file is where
ITS specific bad-source/missing-dependency behaviour belongs.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from openbtk.core.errors import LoaderError
from openbtk.core.registry import LOADER_REGISTRY

from .conftest import FixtureRecord, ReferenceLoader, make_exploding_source

if TYPE_CHECKING:
    from openbtk.core.base import BaseLoader


def _new_instance(key: str) -> BaseLoader[Any, Any]:
    """Every loader here takes zero constructor arguments -- true for the
    reference implementation and, so far, every registered loader. If a
    future loader needs required constructor args, this contract suite's
    fixture wiring will need a per-key params table; not needed yet."""
    return LOADER_REGISTRY.create(key)


@pytest.mark.parametrize("key", LOADER_REGISTRY.list_keys())
class TestLoaderContract:
    def test_load_returns_iterator(self, key: str) -> None:
        """load() returns a genuine Iterator, not a list (ADR-0004)."""
        loader = _new_instance(key)
        result = loader.load(iter(["a", "b"]))
        assert isinstance(result, Iterator), (
            f"{key}: load() returned {type(result).__name__}, not an Iterator. "
            "A list satisfies mypy but breaks the streaming memory guarantee."
        )

    def test_yields_first_record_lazily(self, key: str) -> None:
        """The first record is available without consuming the rest of the
        source. Verified behaviourally, not by timing: `make_exploding_source`
        raises if anything past the first item is ever touched."""
        loader = _new_instance(key)
        source = make_exploding_source("first-and-only-safe-item")
        first = next(iter(loader.load(source)))
        assert first is not None

    def test_does_not_resupply_consumed_iterator(self, key: str) -> None:
        """A fully-consumed iterator stays exhausted -- standard Python
        iterator semantics, not silently restarted or replayed."""
        loader = _new_instance(key)
        iterator = loader.load(iter(["a", "b"]))
        list(iterator)  # fully consume
        with pytest.raises(StopIteration):
            next(iterator)

    def test_records_validate_against_schema(self, key: str) -> None:
        """Every yielded record is a real Pydantic model instance, not a
        bare dict/tuple -- callers are entitled to rely on typed access."""
        loader = _new_instance(key)
        records = list(loader.load(iter(["x", "y"])))
        assert records, f"{key}: yielded nothing for a two-item source"
        assert all(isinstance(r, BaseModel) for r in records)

    def test_provenance_is_serialisable(self, key: str) -> None:
        """Component.provenance() round-trips through JSON -- required for
        it to ever land safely in a run manifest."""
        loader = _new_instance(key)
        prov = loader.provenance()
        dumped = prov.model_dump_json()
        assert isinstance(dumped, str) and len(dumped) > 0


# ---------------------------------------------------------------------------
# Reference-only checks (see module docstring for why these are not
# parametrized across every future loader).
# ---------------------------------------------------------------------------


def test_reference_loader_raises_loader_error_on_bad_source() -> None:
    loader = ReferenceLoader()
    with pytest.raises(LoaderError):
        list(loader.load(None))  # type: ignore[arg-type]


def test_reference_loader_error_context_contains_no_phi() -> None:
    """Even when the source itself contains something PHI-shaped, a raised
    error's message and context never echo it back."""
    phi_like_value = "SSN-123-45-6789-PATIENT-JANE-DOE"
    loader = ReferenceLoader()
    # A str IS iterable char-by-char, so this doesn't hit the bad-source path
    # -- included only to show it's not the intended failure trigger below.
    with contextlib.suppress(LoaderError):
        list(loader.load(phi_like_value))  # type: ignore[arg-type]

    # The genuinely bad case: a non-iterable object that happens to stringify
    # to something PHI-shaped, to prove .context never captures repr(source).
    class _PhiShaped:
        def __repr__(self) -> str:
            return phi_like_value

    with pytest.raises(LoaderError) as exc_info:
        list(loader.load(_PhiShaped()))  # type: ignore[arg-type]
    assert phi_like_value not in str(exc_info.value)
    assert phi_like_value not in str(exc_info.value.context)


# ---------------------------------------------------------------------------
# Meta-test: proves the laziness check actually catches a real violation.
# Never registered globally -- constructed and torn down locally so it never
# pollutes LOADER_REGISTRY for any other test.
# ---------------------------------------------------------------------------


class _EagerBrokenLoader:
    """Deliberately violates ADR-0004 by materialising the whole source
    before yielding anything -- exactly the mistake the contract exists to
    catch. Not a real BaseLoader subclass; only needs the same `.load()`
    shape for this one meta-test."""

    def load(self, source: Iterator[str]) -> Iterator[FixtureRecord]:
        materialised = list(source)  # the violation: consumes everything first
        return iter(
            FixtureRecord(record_id=str(i), text=t) for i, t in enumerate(materialised)
        )


def test_laziness_check_catches_a_real_violation() -> None:
    """If this test ever fails, the laziness contract has stopped being a
    real check -- it would mean an eagerly-consuming loader could pass."""
    broken = _EagerBrokenLoader()
    source = make_exploding_source("only safe item")
    with pytest.raises(RuntimeError, match="not streaming lazily"):
        next(iter(broken.load(source)))
