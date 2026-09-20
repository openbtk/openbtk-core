"""k-anonymity for exported structured data (FR-D-11).

Removing direct identifiers is not enough for a table of patients: a combination of
ordinary attributes (birth year, sex, a ZIP prefix, an admission month) can single a
person out. A table is **k-anonymous** over a set of *quasi-identifiers* when every
combination of their values is shared by at least ``k`` records; ``k = 1`` means
someone is unique and re-identifiable by anyone who knows those attributes.

This module does two things over a plain table (an iterable of mappings), so it works
for any modality's export, not only EHR:

* :func:`k_anonymity_report` measures it. The report holds **counts only**, never a
  value, so it is safe to log or attach to a manifest.
* :func:`anonymise_to_k` reaches it by *generalising* the quasi-identifiers along
  ladders you can inspect (year -> 5-year band -> 10-year band -> ``*``) and, as a last
  resort, *suppressing* the records that are still too rare. This is the Datafly
  heuristic (Sweeney, 2002): generalise the attribute with the most distinct values,
  one level at a time, until the table is k-anonymous or few enough records remain to
  suppress. It is greedy, not optimal: it may generalise more than the minimum, and
  it reports exactly what it did.

What this does not do, and what k-anonymity does not give you:

* It protects against *linking on the columns you named*. A quasi-identifier you did
  not name (a rare diagnosis, a free-text note) is not protected; choosing them is the
  data owner's job, and this module cannot do it for you.
* k-anonymity says nothing about what a group *shares*: if every record in a class has
  the same diagnosis, the diagnosis is disclosed (a homogeneity attack). It is one piece
  of evidence for an Expert Determination, not a determination.
* It does not decide the right ``k``. ``5`` is a common floor; the appropriate value is
  a policy decision for your data and setting.
* The ZIP ladder shortens a code but does not know which three-digit prefixes cover too
  few people to publish (HIPAA Safe Harbor lists them); apply that rule yourself.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field

from openbtk.core.schemas import JsonValue  # noqa: TC001

Generaliser = Callable[[object], Hashable]
"""A function from an original value to a coarser one. ``None`` (unknown) passes
through every built-in until the final ``*`` level."""

SUPPRESSED = "*"
"""What a fully generalised value is written as."""

_MAX_LEVELS = 12
_SMALLEST_CLASSES = 10


# ------------------------------------------------------------------ ladders


def _plain(value: object) -> str | int | float | bool | None:
    """``value`` as a scalar a table cell, a JSON file and a dict key can all hold."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def identity(value: object) -> Hashable:
    """Level 0: the value as it is (dates as ISO strings).

    Example:
        >>> identity("female")
        'female'
    """
    return _plain(value)


def star(value: object) -> Hashable:
    """The top level: every value becomes ``*``, so the attribute stops distinguishing.

    Example:
        >>> star("anything")
        '*'
    """
    del value
    return SUPPRESSED


def top_code(limit: int) -> Generaliser:
    """Write every integer at or above ``limit`` as ``"<limit>+"`` (HIPAA's ``90+``).

    Example:
        >>> top_code(90)(93), top_code(90)(41)
        ('90+', 41)
    """

    def apply(value: object) -> Hashable:
        if isinstance(value, int) and not isinstance(value, bool) and value >= limit:
            return f"{limit}+"
        return _plain(value)

    return apply


def band(width: int, *, top: int | None = None) -> Generaliser:
    """Group integers into bands of ``width`` (``1987`` -> ``"1985-1989"`` for 5).

    Args:
        width: Band width, at least 2.
        top: Values at or above this are written ``"<top>+"``. ``None`` disables it.

    Non-integers (including ``None``) pass through, so an unknown stays unknown rather
    than being placed in a band.

    Example:
        >>> band(5)(1987)
        '1985-1989'
        >>> band(10, top=90)(93)
        '90+'
    """
    if width < 2:
        raise ValueError("band width must be at least 2")

    def apply(value: object) -> Hashable:
        if isinstance(value, bool) or not isinstance(value, int):
            return _plain(value)
        if top is not None and value >= top:
            return f"{top}+"
        low = (value // width) * width
        return f"{low}-{low + width - 1}"

    return apply


def prefix(digits: int) -> Generaliser:
    """Keep the first ``digits`` characters, mask the rest (``"90210"`` -> ``"902**"``).

    Example:
        >>> prefix(3)("90210")
        '902**'
    """
    if digits < 1:
        raise ValueError("digits must be at least 1")

    def apply(value: object) -> Hashable:
        if not isinstance(value, str):
            return _plain(value)
        return value[:digits] + "*" * max(0, len(value) - digits)

    return apply


def zip_ladder(length: int = 5) -> tuple[Generaliser, ...]:
    """A ladder for a ZIP code: the full code, then ever shorter prefixes, then ``*``.

    Example:
        >>> [level("90210") for level in zip_ladder(5)]
        ['90210', '9021*', '902**', '90***', '9****', '*']
    """
    return (identity, *(prefix(d) for d in range(length - 1, 0, -1)), star)


def _year_of(value: object) -> Hashable:
    """``"2024-03"`` -> ``"2024"``; anything else unchanged."""
    if isinstance(value, str) and len(value) >= 4 and value[:4].isdigit():
        return value[:4]
    return _plain(value)


DEFAULT_LADDERS: Mapping[str, tuple[Generaliser, ...]] = {
    "birth_year": (identity, band(5), band(10), band(20), star),
    "age": (top_code(90), band(5, top=90), band(10, top=90), band(20, top=90), star),
    "gender": (identity, star),
    "race": (identity, star),
    "ethnicity": (identity, star),
    "admission_year": (identity, band(5), star),
    "admission_month": (identity, _year_of, star),
}
"""Ladders for the EHR quasi-identifiers. ``age`` starts by top-coding at 90, as HIPAA's
Safe Harbor requires. Pass ``ladders=`` to override or add (a ``zip_ladder()``, say)."""

_FALLBACK_LADDER: tuple[Generaliser, ...] = (identity, star)


# ------------------------------------------------------------------ reports


class KAnonymityReport(BaseModel):
    """How k-anonymous a table is. Counts only: no value from the data appears here.

    Example:
        >>> rows = [{"sex": "f"}, {"sex": "f"}, {"sex": "m"}]
        >>> report = k_anonymity_report(rows, ["sex"], k=2)
        >>> (report.k_achieved, report.n_unique, report.satisfied)
        (1, 1, False)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    k: int = Field(..., ge=1, description="The threshold that was asked for.")
    quasi_identifiers: list[str] = Field(..., description="The columns that were used.")
    n_records: int = Field(..., ge=0)
    n_classes: int = Field(..., ge=0, description="Distinct combinations of values.")
    k_achieved: int = Field(
        ..., ge=0, description="The smallest class size (0 for an empty table)."
    )
    n_unique: int = Field(
        ..., ge=0, description="Records alone in their class: the sharpest risk."
    )
    n_below_k: int = Field(
        ..., ge=0, description="Records in a class smaller than ``k``."
    )
    smallest_classes: list[int] = Field(
        default_factory=list, description="The sizes of the smallest classes."
    )
    satisfied: bool = Field(
        ..., description="Every class has at least ``k`` records (true for no records)."
    )


class AnonymisationResult(BaseModel):
    """The exported table and an account of how it was made.

    ``rows`` holds the generalised quasi-identifiers and any ``keep`` columns, and
    nothing else: columns that were not named are dropped, so an unlisted column cannot
    leak through.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows: list[dict[str, JsonValue]]
    report: KAnonymityReport = Field(
        ..., description="k-anonymity of ``rows`` (after suppression)."
    )
    levels: dict[str, int] = Field(
        ..., description="How many ladder steps each quasi-identifier was raised."
    )
    n_input: int = Field(..., ge=0)
    n_suppressed: int = Field(
        ..., ge=0, description="Records removed because their class stayed below ``k``."
    )


def _key(row: Mapping[str, object], names: Sequence[str]) -> tuple[Hashable, ...]:
    return tuple(_plain(row.get(name)) for name in names)


def _report(
    classes: Counter[tuple[Hashable, ...]], names: Sequence[str], k: int
) -> KAnonymityReport:
    sizes = sorted(classes.values())
    return KAnonymityReport(
        k=k,
        quasi_identifiers=list(names),
        n_records=sum(sizes),
        n_classes=len(sizes),
        k_achieved=sizes[0] if sizes else 0,
        n_unique=sum(1 for s in sizes if s == 1),
        n_below_k=sum(s for s in sizes if s < k),
        smallest_classes=sizes[:_SMALLEST_CLASSES],
        satisfied=all(s >= k for s in sizes),
    )


def k_anonymity_report(
    rows: Iterable[Mapping[str, object]], quasi_identifiers: Sequence[str], *, k: int
) -> KAnonymityReport:
    """Measure k-anonymity of ``rows`` over ``quasi_identifiers``.

    Streams the rows: memory is one counter entry per distinct combination, not one per
    record. A missing value counts as its own value ("unknown"), so records with the
    same gaps are in the same class.

    Args:
        rows: The table.
        quasi_identifiers: The columns an outsider could link on.
        k: The threshold to report against.

    Example:
        >>> rows = [{"sex": "f", "yob": 1980}] * 3 + [{"sex": "m", "yob": 1975}]
        >>> report = k_anonymity_report(rows, ["sex", "yob"], k=3)
        >>> report.k_achieved, report.n_below_k, report.satisfied
        (1, 1, False)
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    names = list(quasi_identifiers)
    if not names:
        raise ValueError("name at least one quasi-identifier")
    return _report(Counter(_key(row, names) for row in rows), names, k)


def anonymise_to_k(
    rows: Iterable[Mapping[str, object]],
    quasi_identifiers: Sequence[str],
    *,
    k: int,
    ladders: Mapping[str, Sequence[Generaliser]] | None = None,
    max_suppression: float = 0.05,
    keep: Sequence[str] = (),
) -> AnonymisationResult:
    """Generalise (and if need be suppress) ``rows`` until they are k-anonymous.

    Args:
        rows: The table. Materialised once, as the algorithm needs every record.
        quasi_identifiers: The columns to generalise.
        k: The class size to reach.
        ladders: Per column, a sequence of levels: level 0 first (usually the value as
            it is), each next one coarser, ending in ``*`` if the column may vanish.
            Columns not given here use :data:`DEFAULT_LADDERS`, then ``value -> *``.
        max_suppression: The fraction of records that may be removed instead of
            generalising further. ``0`` never suppresses until the ladders run out.
        keep: Extra columns copied through untouched (a pseudonymous id, say). Nothing
            else is exported.

    Returns:
        The exported rows and an account of the levels reached and records suppressed.

    Example:
        >>> rows = [{"yob": 1980 + i % 3, "sex": "f"} for i in range(9)]
        >>> result = anonymise_to_k(rows, ["yob", "sex"], k=3)
        >>> result.report.satisfied, result.n_suppressed
        (True, 0)
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if not 0 <= max_suppression <= 1:
        raise ValueError("max_suppression must be between 0 and 1")
    names = list(quasi_identifiers)
    if not names:
        raise ValueError("name at least one quasi-identifier")

    table = [dict(row) for row in rows]
    chosen: dict[str, Sequence[Generaliser]] = {}
    for name in names:
        given = (ladders or {}).get(name)
        ladder = given if given is not None else DEFAULT_LADDERS.get(name)
        ladder = _FALLBACK_LADDER if ladder is None else ladder
        if not 1 <= len(ladder) <= _MAX_LEVELS:
            raise ValueError(
                f"the ladder for {name!r} must have 1-{_MAX_LEVELS} levels"
            )
        chosen[name] = ladder
    level = dict.fromkeys(names, 0)

    def keys() -> list[tuple[Hashable, ...]]:
        return [tuple(chosen[n][level[n]](row.get(n)) for n in names) for row in table]

    current = keys()
    while table:
        classes = Counter(current)
        below = sum(size for size in classes.values() if size < k)
        if below <= max_suppression * len(table):
            break
        movable = [n for n in names if level[n] < len(chosen[n]) - 1]
        if not movable:
            break
        distinct = {
            n: len({chosen[n][level[n]](row.get(n)) for row in table}) for n in movable
        }
        level[max(movable, key=lambda n: (distinct[n], -names.index(n)))] += 1
        current = keys()

    classes = Counter(current)
    kept = [i for i, key in enumerate(current) if classes[key] >= k]
    out: list[dict[str, JsonValue]] = []
    for i in kept:
        exported: dict[str, JsonValue] = {
            n: _plain(chosen[n][level[n]](table[i].get(n))) for n in names
        }
        for extra in keep:
            exported[extra] = _plain(table[i].get(extra))
        out.append(exported)

    return AnonymisationResult(
        rows=out,
        report=_report(Counter(current[i] for i in kept), names, k),
        levels=dict(level),
        n_input=len(table),
        n_suppressed=len(table) - len(kept),
    )
