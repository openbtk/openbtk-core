"""``OMOPLoader``: OMOP CDM v5.4 core tables, read as batched Parquet via
pyarrow (docs/05_DATA_MODALITY_SPEC.md section 2.2).

**Scope and real, disclosed trade-offs** (the same discipline as every other
loader's own docstring in this project):

* **Parquet only.** OMOP CDM is also commonly distributed as CSV or in a
  SQL database; this loader reads neither. Parquet is the one pyarrow's
  own batched, columnar reader (``ParquetFile.iter_batches``) genuinely
  streams from -- the roadmap task names "pyarrow batches" specifically,
  not "any OMOP source."
* **No vocabulary resolution.** A real OMOP CDM stores clinical codes as
  an integer ``*_concept_id`` into the standard OHDSI vocabulary, which
  requires the (multi-gigabyte, separately licensed-ish) ``concept``
  table to resolve to a real ``(system, code)`` pair -- OpenBTK does not
  bundle it (docs/09_CODING_STANDARDS.md section 7; that resolution
  belongs in ``openbtk.terminology``, unbuilt, M7). This loader reads the
  ``*_source_value`` column instead -- the original source-vocabulary
  code CDM ETL processes are required to preserve alongside the
  standardised concept -- and lets the caller declare which
  :class:`~openbtk.core.schemas.CodeSystem` each table's source values
  are actually in via constructor arguments (defaults reflect the most
  common real convention: SNOMED for conditions, RxNorm for drugs, CPT
  for procedures, LOINC for measurements), since that is genuinely
  site-specific and this loader cannot know it from the data alone.
* **Memory is not O(batch) across the whole loader**, unlike
  ``PlainTextLoader``/``MIMICNotesLoader``. OMOP's normalised,
  multi-table schema does not co-locate one patient's events -- a
  condition for person 1 could be the very last row of
  ``condition_occurrence.parquet``. Producing a complete
  ``PatientRecord`` therefore requires each event table to be fully
  scanned (in real pyarrow batches, never one big ``read_table()`` call,
  so peak memory per *table scan* stays bounded) before the first record
  can be assembled and yielded; total memory scales with the corpus's
  event count, not the batch size. Disclosed here rather than claimed
  away -- this is a real, structural difference between a flat
  per-record format and a relational join, not a bug to silently fix.
* ``death`` (OMOP's dedicated mortality table) is not read -- every
  ``Demographics.deceased`` is ``False`` here. A real deployment's
  ``person`` table alone never carries death status in OMOP CDM.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from openbtk.core._lazy import require
from openbtk.core.base import BaseLoader
from openbtk.core.errors import LoaderError
from openbtk.core.logging import get_logger
from openbtk.core.registry import LOADER_REGISTRY
from openbtk.core.schemas import CodeSystem
from openbtk.data.ehr.schemas import (
    CodedEvent,
    Demographics,
    Encounter,
    Measurement,
    PatientRecord,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import ModuleType

log = get_logger(__name__)

_SOURCE_SYSTEM = "omop-cdm-5.4"

# The OMOP standard vocabulary's own fixed "Gender" concept ids -- stable
# since OMOP CDM's inception, not fabricated. Any other value (a
# vocabulary-specific or unmapped concept) is reported as None rather than
# guessed, since resolving it for real needs the concept table this loader
# deliberately does not bundle (see module docstring).
_OMOP_GENDER_CONCEPT_TO_GENDER: dict[int, Literal["male", "female"]] = {
    8507: "male",
    8532: "female",
}


def _coerce_datetime(value: Any) -> datetime | None:
    """OMOP CDM records no time zone at all; pyarrow returns naive
    ``datetime``/``date`` values for every Parquet timestamp/date column.
    Treating them as UTC is a documented, disclosed convention -- not a
    discovered fact about the source data -- so every ``Encounter``/
    ``CodedEvent``/``Measurement`` timestamp built from OMOP data still
    satisfies the project-wide timezone-aware requirement."""
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    return None


def _birth_date_from_row(row: dict[str, Any]) -> date | None:
    birth_datetime = row.get("birth_datetime")
    if isinstance(birth_datetime, datetime):
        return birth_datetime.date()
    if isinstance(birth_datetime, date):
        return birth_datetime
    year = row.get("year_of_birth")
    if year is None:
        return None
    month = row.get("month_of_birth") or 1
    day = row.get("day_of_birth") or 1
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        # A real, malformed day-of-month for the given month (e.g. day=31
        # in February) -- degrade to the first of the month rather than
        # raising, since the year itself (the part cohort-building actually
        # depends on for age_between()) is still known and usable.
        return date(int(year), int(month), 1)


def _gender_from_concept_id(concept_id: Any) -> Literal["male", "female"] | None:
    if concept_id is None:
        return None
    return _OMOP_GENDER_CONCEPT_TO_GENDER.get(int(concept_id))


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _visit_ref(visit_occurrence_id: Any) -> str | None:
    return None if visit_occurrence_id is None else str(visit_occurrence_id)


@LOADER_REGISTRY.register("loader.ehr.omop")
class OMOPLoader(BaseLoader[str, PatientRecord]):
    """Load ``PatientRecord``s from a directory of OMOP CDM v5.4 Parquet
    tables (``person.parquet`` required; ``visit_occurrence``,
    ``condition_occurrence``, ``drug_exposure``, ``procedure_occurrence``
    and ``measurement`` are each read if present, skipped if absent). See
    this module's own docstring for real, disclosed scope and memory
    trade-offs.

    Example:
        >>> import contextlib, io, tempfile
        >>> from pathlib import Path
        >>> import pyarrow as pa
        >>> import pyarrow.parquet as pq
        >>> with tempfile.TemporaryDirectory() as d:
        ...     table = pa.table({
        ...         "person_id": [1],
        ...         "gender_concept_id": [8532],
        ...         "year_of_birth": [1980],
        ...         "month_of_birth": [5],
        ...         "day_of_birth": [1],
        ...     })
        ...     _ = pq.write_table(table, Path(d) / "person.parquet")
        ...     with contextlib.redirect_stdout(io.StringIO()):
        ...         records = list(OMOPLoader().load(d))
        >>> records[0].patient_id
        '1'
        >>> records[0].demographics.gender
        'female'
    """

    def __init__(
        self,
        *,
        batch_size: int = 10_000,
        condition_system: CodeSystem = CodeSystem.SNOMED,
        drug_system: CodeSystem = CodeSystem.RXNORM,
        procedure_system: CodeSystem = CodeSystem.CPT,
        measurement_system: CodeSystem = CodeSystem.LOINC,
    ) -> None:
        self._batch_size = batch_size
        self._condition_system = condition_system
        self._drug_system = drug_system
        self._procedure_system = procedure_system
        self._measurement_system = measurement_system

    def load(self, source: str) -> Iterator[PatientRecord]:
        """Raises:
        LoaderError: If ``source`` is not a directory, lacks
            ``person.parquet``, or a Parquet table fails to parse.
        """
        directory = Path(source)
        if not directory.is_dir():
            raise LoaderError(
                f"Not a directory: {source}",
                context={"modality": "ehr", "stage": "load"},
            )
        person_path = directory / "person.parquet"
        if not person_path.is_file():
            raise LoaderError(
                f"{source} is missing the required person.parquet table.",
                context={"modality": "ehr", "stage": "load"},
            )
        pq = require("pyarrow.parquet", extra="ehr")
        log.info("loader.start", modality="ehr", source=str(directory))

        encounters_by_person = self._load_visits(directory, pq)
        conditions_by_person = self._load_events(
            directory, "condition_occurrence.parquet", pq, self._condition_row_to_event
        )
        medications_by_person = self._load_events(
            directory, "drug_exposure.parquet", pq, self._drug_row_to_event
        )
        procedures_by_person = self._load_events(
            directory, "procedure_occurrence.parquet", pq, self._procedure_row_to_event
        )
        observations_by_person = self._load_measurements(directory, pq)

        for row in self._iter_rows(person_path, pq):
            person_id = str(row["person_id"])
            yield PatientRecord(
                patient_id=person_id,
                demographics=self._demographics_from_row(row),
                encounters=encounters_by_person.get(person_id, []),
                conditions=conditions_by_person.get(person_id, []),
                medications=medications_by_person.get(person_id, []),
                procedures=procedures_by_person.get(person_id, []),
                observations=observations_by_person.get(person_id, []),
                source_system=_SOURCE_SYSTEM,
            )

    def _iter_rows(self, path: Path, pq: ModuleType) -> Iterator[dict[str, Any]]:
        try:
            parquet_file = pq.ParquetFile(path)
        except Exception as e:
            raise LoaderError(
                f"Failed to read {path.name}.",
                context={"modality": "ehr", "stage": "load", "filename": path.name},
            ) from e
        for batch in parquet_file.iter_batches(batch_size=self._batch_size):
            yield from batch.to_pylist()

    def _demographics_from_row(self, row: dict[str, Any]) -> Demographics:
        return Demographics(
            birth_date=_birth_date_from_row(row),
            gender=_gender_from_concept_id(row.get("gender_concept_id")),
            race=_optional_str(row.get("race_source_value")),
            ethnicity=_optional_str(row.get("ethnicity_source_value")),
        )

    def _load_visits(
        self, directory: Path, pq: ModuleType
    ) -> dict[str, list[Encounter]]:
        encounters_by_person: dict[str, list[Encounter]] = {}
        path = directory / "visit_occurrence.parquet"
        if not path.is_file():
            return encounters_by_person
        for row in self._iter_rows(path, pq):
            person_id = str(row["person_id"])
            visit_id = str(row["visit_occurrence_id"])
            encounters_by_person.setdefault(person_id, []).append(
                Encounter(
                    encounter_id=visit_id,
                    encounter_type=_optional_str(row.get("visit_source_value")),
                    start=_coerce_datetime(
                        row.get("visit_start_datetime") or row.get("visit_start_date")
                    ),
                    end=_coerce_datetime(
                        row.get("visit_end_datetime") or row.get("visit_end_date")
                    ),
                    status=None,
                )
            )
        return encounters_by_person

    def _load_events(
        self,
        directory: Path,
        filename: str,
        pq: ModuleType,
        row_to_event: Callable[[dict[str, Any]], CodedEvent | None],
    ) -> dict[str, list[CodedEvent]]:
        events_by_person: dict[str, list[CodedEvent]] = {}
        path = directory / filename
        if not path.is_file():
            return events_by_person
        for row in self._iter_rows(path, pq):
            event = row_to_event(row)
            if event is None:
                continue
            person_id = str(row["person_id"])
            events_by_person.setdefault(person_id, []).append(event)
        return events_by_person

    def _condition_row_to_event(self, row: dict[str, Any]) -> CodedEvent | None:
        code = row.get("condition_source_value")
        if not code:
            return None
        return CodedEvent(
            code=str(code),
            system=self._condition_system,
            timestamp=_coerce_datetime(
                row.get("condition_start_datetime") or row.get("condition_start_date")
            ),
            encounter_ref=_visit_ref(row.get("visit_occurrence_id")),
            status=_optional_str(row.get("condition_status_source_value")),
        )

    def _drug_row_to_event(self, row: dict[str, Any]) -> CodedEvent | None:
        code = row.get("drug_source_value")
        if not code:
            return None
        return CodedEvent(
            code=str(code),
            system=self._drug_system,
            timestamp=_coerce_datetime(
                row.get("drug_exposure_start_datetime")
                or row.get("drug_exposure_start_date")
            ),
            encounter_ref=_visit_ref(row.get("visit_occurrence_id")),
        )

    def _procedure_row_to_event(self, row: dict[str, Any]) -> CodedEvent | None:
        code = row.get("procedure_source_value")
        if not code:
            return None
        return CodedEvent(
            code=str(code),
            system=self._procedure_system,
            timestamp=_coerce_datetime(
                row.get("procedure_datetime") or row.get("procedure_date")
            ),
            encounter_ref=_visit_ref(row.get("visit_occurrence_id")),
        )

    def _load_measurements(
        self, directory: Path, pq: ModuleType
    ) -> dict[str, list[Measurement]]:
        observations_by_person: dict[str, list[Measurement]] = {}
        path = directory / "measurement.parquet"
        if not path.is_file():
            return observations_by_person
        for row in self._iter_rows(path, pq):
            code = row.get("measurement_source_value")
            if not code:
                continue
            person_id = str(row["person_id"])
            value: float | str | None = row.get("value_as_number")
            if value is None:
                value = _optional_str(row.get("value_source_value"))
            reference_range = None
            low, high = row.get("range_low"), row.get("range_high")
            if low is not None and high is not None:
                reference_range = (float(low), float(high))
            observations_by_person.setdefault(person_id, []).append(
                Measurement(
                    code=str(code),
                    system=self._measurement_system,
                    value=value,
                    unit=_optional_str(row.get("unit_source_value")),
                    reference_range=reference_range,
                    timestamp=_coerce_datetime(
                        row.get("measurement_datetime") or row.get("measurement_date")
                    ),
                    encounter_ref=_visit_ref(row.get("visit_occurrence_id")),
                )
            )
        return observations_by_person
