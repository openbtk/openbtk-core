"""Shared contract every registered BaseLoader implementation must satisfy.

Parametrized over LOADER_REGISTRY.list_keys() (docs/07_TEST_CHARTER.md
section 3.1): registering a loader anywhere automatically enrolls it here,
with no opt-out. Beyond tests/contract/conftest.py's ReferenceLoader (whose
``InputT`` is ``Iterator[str]``), M3 added the first real modality
loaders -- and they take a file/directory *path* (``str``), a genuinely
different ``InputT`` shape. ``_make_source`` below is exactly the per-key
fixture wiring this module's own docstring already anticipated for
constructor args ("not needed yet") -- now needed, and for sources too.

Three checks from the doc's illustrative template
(test_raises_loader_error_on_bad_source, test_error_context_contains_no_phi,
test_missing_dependency_names_the_extra) are NOT generically parametrizable
across an arbitrary future loader: what counts as "a bad source" is
inherently loader-specific, and a loader with zero optional dependencies has
no missing-dependency path to test at all. Two more
(test_yields_first_record_lazily, test_does_not_resupply_consumed_iterator)
are specifically about Python *iterator protocol* semantics
(``StopIteration``, an exploding generator) that only mean something for a
source that genuinely is an iterator -- a file path is not one (confirmed:
``isinstance("a/path", Iterator)`` is ``False``, since ``str`` has no
``__next__``), so these two skip for a path-shaped source rather than being
forced into a shape that would not actually test anything real. A future
modality's own test file is where ITS specific bad-source,
missing-dependency, and streaming-without-materialising-the-whole-file
behaviour belongs (see tests/unit/data/clinical_text/test_loaders.py).
"""

from __future__ import annotations

import contextlib
import importlib.util
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from openbtk.core.errors import LoaderError
from openbtk.core.registry import LOADER_REGISTRY

from .conftest import FixtureRecord, ReferenceLoader, enrolled, make_exploding_source

if TYPE_CHECKING:
    from pathlib import Path

    from openbtk.core.base import BaseLoader

# CI's own test-core job installs with zero optional extras (NFR-10) --
# a loader whose .load() genuinely needs one (MIMICNotesLoader needs
# pandas) cannot be driven through this generic suite there. Skipped per
# key rather than assumed always installed; found by actually reproducing
# a genuinely clean venv, not assumed from the lazy-import design alone.
_MISSING_DEPENDENCY_BY_KEY: dict[str, str] = {}
if importlib.util.find_spec("pandas") is None:
    _MISSING_DEPENDENCY_BY_KEY["loader.clinical_text.mimic_notes"] = "pandas"
if importlib.util.find_spec("fhir") is None:
    _MISSING_DEPENDENCY_BY_KEY["loader.ehr.fhir"] = "fhir.resources"
if importlib.util.find_spec("pyarrow") is None:
    _MISSING_DEPENDENCY_BY_KEY["loader.ehr.omop"] = "pyarrow"
if importlib.util.find_spec("hl7apy") is None:
    _MISSING_DEPENDENCY_BY_KEY["loader.ehr.hl7v2"] = "hl7apy"


def _skip_if_missing_dependency(key: str) -> None:
    dependency = _MISSING_DEPENDENCY_BY_KEY.get(key)
    if dependency is not None:
        pytest.skip(f"{key}: requires the 'text' extra ({dependency})")


def _new_instance(key: str) -> BaseLoader[Any, Any]:
    """Every loader here takes zero constructor arguments -- true for the
    reference implementation and every registered loader so far. If a
    future loader needs required constructor args, this contract suite's
    fixture wiring will need a per-key params table too."""
    return LOADER_REGISTRY.create(key)


def _make_source(key: str, tmp_path: Path) -> Any:
    """A valid, minimal, real two-record source for ``key``. Falls back to
    the original in-memory iterator shape for any key not listed here --
    add a case the moment a new loader's InputT needs one, per this
    module's own established "not needed yet" philosophy."""
    if key == "loader.clinical_text.plain_text":
        (tmp_path / "a.txt").write_text("first", encoding="utf-8")
        (tmp_path / "b.txt").write_text("second", encoding="utf-8")
        return str(tmp_path)
    if key == "loader.clinical_text.jsonl":
        path = tmp_path / "notes.jsonl"
        path.write_text(
            '{"record_id": "n1", "text": "first"}\n'
            '{"record_id": "n2", "text": "second"}\n',
            encoding="utf-8",
        )
        return str(path)
    if key == "loader.clinical_text.mimic_notes":
        path = tmp_path / "notes.csv"
        path.write_text("ROW_ID,TEXT\n1,first\n2,second\n", encoding="utf-8")
        return str(path)
    if key == "loader.ehr.fhir":
        # A Bundle is plain JSON -- writing one needs no fhir.resources
        # import at all (only FHIRLoader.load() itself does, lazily); two
        # single-Patient bundles are a valid, minimal, real two-record source.
        import json

        for patient_id in ("pt-1", "pt-2"):
            bundle = {
                "resourceType": "Bundle",
                "type": "collection",
                "entry": [{"resource": {"resourceType": "Patient", "id": patient_id}}],
            }
            (tmp_path / f"{patient_id}.json").write_text(
                json.dumps(bundle), encoding="utf-8"
            )
        return str(tmp_path)
    if key == "loader.ehr.hl7v2":
        # HL7 v2 is plain text -- writing a message needs no hl7apy at all. Two
        # ADT messages about two patients in one file: a valid two-record source.
        for patient_id in ("PT1", "PT2"):
            message = "\r".join(
                [
                    "MSH|^~\\&|APP|FAC|RCV|FAC|20240314101500||ADT^A01|M1|P|2.5",
                    f"PID|1||{patient_id}^^^HOSP^MR||DOE^JANE||19800601|F",
                ]
            )
            (tmp_path / f"{patient_id}.hl7").write_text(message, encoding="utf-8")
        return str(tmp_path)
    if key == "loader.ehr.omop":
        # Unlike the JSON case above, a real Parquet file genuinely needs
        # pyarrow to write -- guarded so the two iterator-protocol checks
        # below (which call this function unconditionally, before any
        # _skip_if_missing_dependency gate) never raise ImportError in a
        # zero-extras environment; they only inspect this return value's
        # type and never actually call .load() on it when pyarrow is absent.
        if importlib.util.find_spec("pyarrow") is None:
            return str(tmp_path)
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(
            pa.table({"person_id": [1, 2], "gender_concept_id": [8532, 8507]}),
            tmp_path / "person.parquet",
        )
        return str(tmp_path)
    return iter(["a", "b"])


@pytest.mark.parametrize("key", enrolled(LOADER_REGISTRY))
class TestLoaderContract:
    def test_load_returns_iterator(self, key: str, tmp_path: Path) -> None:
        """load() returns a genuine Iterator, not a list (ADR-0004)."""
        _skip_if_missing_dependency(key)
        loader = _new_instance(key)
        result = loader.load(_make_source(key, tmp_path))
        assert isinstance(result, Iterator), (
            f"{key}: load() returned {type(result).__name__}, not an Iterator. "
            "A list satisfies mypy but breaks the streaming memory guarantee."
        )

    def test_yields_first_record_lazily(self, key: str, tmp_path: Path) -> None:
        """The first record is available without consuming the rest of the
        source. Verified behaviourally, not by timing: `make_exploding_source`
        raises if anything past the first item is ever touched.

        Only meaningful for a genuinely iterator-shaped source -- skipped
        for a path-shaped one (see module docstring)."""
        loader = _new_instance(key)
        if not isinstance(_make_source(key, tmp_path), Iterator):
            pytest.skip(f"{key}: source is not iterator-shaped")
        source = make_exploding_source("first-and-only-safe-item")
        first = next(iter(loader.load(source)))
        assert first is not None

    def test_does_not_resupply_consumed_iterator(
        self, key: str, tmp_path: Path
    ) -> None:
        """A fully-consumed iterator stays exhausted -- standard Python
        iterator semantics, not silently restarted or replayed. Only
        meaningful for a genuinely iterator-shaped source."""
        loader = _new_instance(key)
        if not isinstance(_make_source(key, tmp_path), Iterator):
            pytest.skip(f"{key}: source is not iterator-shaped")
        iterator = loader.load(iter(["a", "b"]))
        list(iterator)  # fully consume
        with pytest.raises(StopIteration):
            next(iterator)

    def test_records_validate_against_schema(self, key: str, tmp_path: Path) -> None:
        """Every yielded record is a real Pydantic model instance, not a
        bare dict/tuple -- callers are entitled to rely on typed access."""
        _skip_if_missing_dependency(key)
        loader = _new_instance(key)
        records = list(loader.load(_make_source(key, tmp_path)))
        assert records, f"{key}: yielded nothing for a two-item source"
        assert all(isinstance(r, BaseModel) for r in records)

    def test_load_all_matches_load(self, key: str, tmp_path: Path) -> None:
        """The base class's load_all() default (list(self.load(source)))
        must agree with load() itself -- it is a documented memory hazard,
        not an alternate code path with its own semantics."""
        _skip_if_missing_dependency(key)
        loader = _new_instance(key)
        # Two independent, identical sources: an iterator-shaped source is
        # single-use by nature, so _make_source is called twice to get two
        # fresh ones. A path-shaped source is safely reusable -- calling it
        # twice with the same tmp_path just rewrites identical file
        # content, which is harmless.
        via_load = list(loader.load(_make_source(key, tmp_path)))
        via_load_all = loader.load_all(_make_source(key, tmp_path))
        assert via_load_all == via_load

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
    phi_like_value = "SSN-123-45-6789-PATIENT-JANE-DOE"  # phi-fixture-ok: adversarial
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
