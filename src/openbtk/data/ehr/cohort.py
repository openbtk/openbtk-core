"""``CohortBuilder``: composable, streaming patient cohort selection
(docs/05_DATA_MODALITY_SPEC.md section 2.4).

**Not a registered pipeline step** -- the spec's own component table
already says so explicitly ("-- (not a pipeline step)"), for the same
underlying reason ``PatientTimelineSerializer`` (task 6.5) is not one
either: composing arbitrary boolean predicates over a whole
``PatientRecord`` is not a ``BaseLoader``/``BasePreprocessor``/
``BaseChunker``/``BaseSegmenter`` shape at all, so there is nothing in
``core.base`` this belongs to.

**Scope, disclosed rather than silently assumed:** docs/05_DATA_MODALITY_SPEC.md
section 2.4's own worked example uses ``before_index=True`` and
``.within(encounter_window(days=30))`` -- temporal predicates relative to
an implicit per-patient "index date" (the date some qualifying event first
occurred). That concept is real in cohort epidemiology, but the spec never
defines how the index date itself is chosen (first qualifying condition?
first encounter? something else?), and inventing an answer here would be
exactly the kind of unverified, unbenchmarked design decision CLAUDE.md
rule 14 exists to prevent. This module ships the composable
``include()``/``exclude()`` builder plus four real, unambiguous predicate
factories (``has_condition``, ``has_medication``, ``has_procedure``,
``age_between``) instead -- genuinely useful and fully tested, without
pretending to solve index-date semantics no one has actually decided on
yet. A follow-up ADR is the right place to define that concept properly,
once a real use case needs it. docs/05_DATA_MODALITY_SPEC.md section 2.4's
worked example is updated in the same change to show only what is built.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from openbtk.data.ehr.schemas import PatientRecord

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from openbtk.core.schemas import CodeSystem
    from openbtk.data.ehr.schemas import CodedEvent

Predicate = Callable[[PatientRecord], bool]
"""A composable cohort-inclusion/exclusion test over one patient."""


def _matches_coded_event(
    events: list[CodedEvent], code: str, system: CodeSystem | None
) -> bool:
    for event in events:
        if event.code != code:
            continue
        if system is not None and event.system is not system:
            continue
        return True
    return False


def has_condition(code: str, *, system: CodeSystem | None = None) -> Predicate:
    """A predicate matching patients with a condition of ``code`` (any
    coding system, unless ``system`` narrows it)."""

    def predicate(patient: PatientRecord) -> bool:
        return _matches_coded_event(patient.conditions, code, system)

    return predicate


def has_medication(code: str, *, system: CodeSystem | None = None) -> Predicate:
    """See :func:`has_condition`; matches ``patient.medications``."""

    def predicate(patient: PatientRecord) -> bool:
        return _matches_coded_event(patient.medications, code, system)

    return predicate


def has_procedure(code: str, *, system: CodeSystem | None = None) -> Predicate:
    """See :func:`has_condition`; matches ``patient.procedures``."""

    def predicate(patient: PatientRecord) -> bool:
        return _matches_coded_event(patient.procedures, code, system)

    return predicate


def age_between(min_age: int, max_age: int, *, as_of: date | None = None) -> Predicate:
    """A predicate matching patients whose age -- computed from
    ``Demographics.birth_date`` as of ``as_of`` -- falls in
    ``[min_age, max_age]`` inclusive.

    ``as_of`` defaults to ``date.today()`` evaluated the moment the
    returned predicate is *called*, not when this factory is, so a
    predicate built once and reused across a long-running process stays
    correct. A patient with no known ``birth_date`` never matches (age is
    genuinely unknown, not assumed to be anything).
    """

    def predicate(patient: PatientRecord) -> bool:
        birth_date = patient.demographics.birth_date
        if birth_date is None:
            return False
        reference = as_of if as_of is not None else datetime.now(tz=UTC).date()
        had_birthday_this_year = (reference.month, reference.day) >= (
            birth_date.month,
            birth_date.day,
        )
        age = reference.year - birth_date.year - (0 if had_birthday_this_year else 1)
        return min_age <= age <= max_age

    return predicate


class CohortBuilder:
    """Compose ``include``/``exclude`` predicates over a stream of
    ``PatientRecord``s -- never materialises the source (ADR-0004): a
    cohort over a million patients has bounded memory, the same guarantee
    every other streaming component in this project makes.

    Example:
        >>> from openbtk.core.schemas import CodeSystem
        >>> from openbtk.data.ehr.schemas import CodedEvent, Demographics
        >>> patients = [
        ...     PatientRecord(
        ...         patient_id="pt-1", demographics=Demographics(),
        ...         conditions=[CodedEvent(code="385093006", system=CodeSystem.SNOMED)],
        ...         source_system="fhir-r4",
        ...     ),
        ...     PatientRecord(
        ...         patient_id="pt-2",
        ...         demographics=Demographics(),
        ...         source_system="fhir-r4",
        ...     ),
        ... ]
        >>> cohort = CohortBuilder(patients).include(has_condition("385093006"))
        >>> [p.patient_id for p in cohort] == ["pt-1"]
        True
    """

    def __init__(self, records: Iterable[PatientRecord]) -> None:
        self._records = records
        self._includes: list[Predicate] = []
        self._excludes: list[Predicate] = []

    def include(self, predicate: Predicate) -> CohortBuilder:
        """Add a predicate every candidate must satisfy (logical AND
        across every ``include()`` call). Returns ``self`` for chaining."""
        self._includes.append(predicate)
        return self

    def exclude(self, predicate: Predicate) -> CohortBuilder:
        """Add a predicate that, if satisfied, drops a candidate (logical
        OR across every ``exclude()`` call). Returns ``self`` for
        chaining."""
        self._excludes.append(predicate)
        return self

    def __iter__(self) -> Iterator[PatientRecord]:
        for record in self._records:
            if all(p(record) for p in self._includes) and not any(
                p(record) for p in self._excludes
            ):
                yield record
