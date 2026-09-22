"""``DosePlausibilityGuardrail``: flags a drug dose, daily total or route in generated
text that falls outside a reference table the caller supplies (FR-G-05).

**OpenBTK ships no dose limits, on purpose.** A table of "safe" doses recalled from
memory, or copied without its source, is a patient-safety hazard: it looks
authoritative and nobody can audit it. So this guardrail is the *engine* -- it finds
dose statements and compares them to limits **you** provide, from a formulary,
guideline or product label you trust. Every :class:`DoseLimit` must name its
``source``, so the provenance of a clinical number is written down where it is used.
The run manifest records a hash of the table, never its content.

What it does, and does not do:

* It checks only the drugs named in the table. A drug that is not in the table is
  not checked, and the result says so; it cannot know a drug it was not told about.
* It reads a dose (``500 mg``, ``0.5 g``, ``5-10 mg``), an optional frequency
  (``BID``, ``every 8 hours``, ``twice daily``) and an optional route (``PO``,
  ``IV``) from the same clause as the drug name. It converts mass units.
* It is a *plausibility* check on wording, not a clinical decision-support system: it
  does not know the patient's weight, renal function or interactions, and it cannot
  read a dose written in words or a table. A pass means "no reference limit was
  broken by a statement I could read", never "this dose is safe".
* An over-the-limit single dose or daily total is ``BLOCK``; a below-minimum dose or
  an unlisted route is ``WARNING``.
* With no reference configured it says that nothing was checked
  (``passed=False``, ``WARNING``); it never reports a silent pass.

All scanning is over bounded windows, so the cost is linear in the text length.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from openbtk.core.base import BaseGuardrail
from openbtk.core.errors import ConfigError
from openbtk.core.registry import GUARDRAIL_REGISTRY
from openbtk.core.schemas import (
    GuardrailResult,
    GuardrailSeverity,
    JsonValue,
    TextSpan,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from openbtk.core.provenance import ComponentProvenance

_MG_PER_UNIT = {
    "g": 1000.0,
    "mg": 1.0,
    "mcg": 0.001,
    "ug": 0.001,
    "µg": 0.001,  # micro sign
    "μg": 0.001,  # Greek mu
}
"""Mass units understood, as milligrams per unit. Other units (mL, units, IU) are not
convertible without a concentration, so a limit in one is compared only to a dose
written in the same unit."""

_NUMBER = r"\d{1,6}(?:\.\d{1,4})?"
_DOSE = re.compile(
    r"(?P<low>"
    + _NUMBER
    + r")(?:\s?(?:-|\N{EN DASH}|to)\s?(?P<high>"
    + _NUMBER
    + r"))?\s?"
    r"(?P<unit>mg|mcg|µg|μg|ug|g|ml|units?|iu)\b",
    re.IGNORECASE,
)
_ROUTE = re.compile(
    r"\b(?P<route>po|oral(?:ly)?|iv|intravenous(?:ly)?|im|intramuscular(?:ly)?|"
    r"sc|subq|subcut(?:aneous(?:ly)?)?|sl|sublingual(?:ly)?|pr|rectal(?:ly)?|"
    r"topical(?:ly)?|inhaled?|inh)\b",
    re.IGNORECASE,
)
_ROUTE_CANONICAL = {
    "po": "oral",
    "oral": "oral",
    "orally": "oral",
    "iv": "intravenous",
    "intravenous": "intravenous",
    "intravenously": "intravenous",
    "im": "intramuscular",
    "intramuscular": "intramuscular",
    "intramuscularly": "intramuscular",
    "sc": "subcutaneous",
    "subq": "subcutaneous",
    "subcut": "subcutaneous",
    "subcutaneous": "subcutaneous",
    "subcutaneously": "subcutaneous",
    "sl": "sublingual",
    "sublingual": "sublingual",
    "sublingually": "sublingual",
    "pr": "rectal",
    "rectal": "rectal",
    "rectally": "rectal",
    "topical": "topical",
    "topically": "topical",
    "inhaled": "inhaled",
    "inhale": "inhaled",
    "inh": "inhaled",
}
_FREQUENCY = re.compile(
    r"\b(?:(?P<fixed>qd|od|daily|once\s+(?:a\s+day|daily)|bid|twice\s+(?:a\s+day|daily)|"
    r"tid|three\s+times\s+(?:a\s+day|daily)|qid|four\s+times\s+(?:a\s+day|daily))|"
    r"q(?P<qh>\d{1,2})h|every\s+(?P<every>\d{1,2})\s+hours?)\b",
    re.IGNORECASE,
)
_PER_DAY = {
    "qd": 1,
    "od": 1,
    "daily": 1,
    "once a day": 1,
    "once daily": 1,
    "bid": 2,
    "twice a day": 2,
    "twice daily": 2,
    "tid": 3,
    "three times a day": 3,
    "three times daily": 3,
    "qid": 4,
    "four times a day": 4,
    "four times daily": 4,
}
_CLAUSE_END = re.compile(r"\.(?!\d)|[;\n]")
"""A period ends a clause unless a digit follows it (the decimal point in ``0.5 g``)."""
_WINDOW = 120
"""How far, in characters, to look for a dose, frequency or route on either side of a
drug name, never past the end of the clause."""


class DoseLimit(BaseModel):
    """One drug's reference limits, with the source they came from.

    Example:
        >>> limit = DoseLimit(
        ...     drug="examplamine", unit="mg", max_single=100,
        ...     source="illustrative values for documentation only",
        ... )
        >>> limit.max_daily is None
        True
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    drug: str = Field(
        ..., min_length=1, description="The drug's name as it should appear in results."
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Other names to match (brand names, abbreviations).",
    )
    unit: str = Field(
        "mg",
        min_length=1,
        description=(
            "The unit the limits are written in (``mg``, ``g``, ``mcg``, ``mL``, "
            "``units``, ...)."
        ),
    )
    min_single: float | None = Field(
        None, ge=0, description="The lowest plausible single dose, if any."
    )
    max_single: float | None = Field(
        None, ge=0, description="The highest single dose, if any."
    )
    max_daily: float | None = Field(
        None,
        ge=0,
        description=(
            "The highest total in a day, if any. Checked only when the text gives a "
            "frequency."
        ),
    )
    routes: list[str] = Field(
        default_factory=list,
        description=(
            "Routes this drug may be given by (``oral``, ``intravenous``, "
            "``intramuscular``, ``subcutaneous``, ``sublingual``, ``rectal``, "
            "``topical``, ``inhaled``). Empty means the route is not checked."
        ),
    )
    source: str = Field(
        ...,
        min_length=1,
        description=(
            "Where these limits come from (a formulary, label or guideline, with "
            "its version). Required: a limit nobody can trace is not usable."
        ),
    )

    @model_validator(mode="after")
    def _limits_are_consistent(self) -> DoseLimit:
        if (
            self.min_single is not None
            and self.max_single is not None
            and self.min_single > self.max_single
        ):
            raise ValueError("min_single must not exceed max_single")
        bad = [r for r in self.routes if r not in set(_ROUTE_CANONICAL.values())]
        if bad:
            raise ValueError(f"unknown route(s) {bad}")
        return self


def _to_unit(value: float, from_unit: str, to_unit: str) -> float | None:
    """``value`` in ``to_unit``, or ``None`` when the units cannot be compared."""
    a, b = from_unit.lower(), to_unit.lower()
    if a == b:
        return value
    if a in _MG_PER_UNIT and b in _MG_PER_UNIT:
        return value * _MG_PER_UNIT[a] / _MG_PER_UNIT[b]
    return None


def _per_day(match: re.Match[str]) -> int | None:
    if match.group("fixed"):
        return _PER_DAY.get(" ".join(match.group("fixed").lower().split()))
    hours = match.group("qh") or match.group("every")
    if hours and int(hours) > 0:
        return max(1, round(24 / int(hours)))
    return None


@GUARDRAIL_REGISTRY.register("guardrail.general.dose_plausibility")
class DosePlausibilityGuardrail(BaseGuardrail):
    """Flags a dose, daily total or route that breaks a limit in a reference table.

    Args:
        reference: Path to a JSON file holding a list of :class:`DoseLimit` objects.
            Read on first use, not at construction.
        limits: The same limits given directly, instead of (or as well as) a file.

    Example:
        >>> limits = [DoseLimit(
        ...     drug="examplamine", unit="mg", max_single=100, max_daily=300,
        ...     source="illustrative values for documentation only",
        ... )]
        >>> guardrail = DosePlausibilityGuardrail(limits=limits)
        >>> guardrail.check("Give examplamine 500 mg PO BID.").passed
        False
        >>> guardrail.check("Give examplamine 50 mg PO BID.").passed
        True
    """

    def __init__(
        self,
        *,
        reference: str | None = None,
        limits: Sequence[DoseLimit | Mapping[str, object]] | None = None,
    ) -> None:
        self._reference = reference
        self._inline = list(limits) if limits else []
        self._loaded: list[DoseLimit] | None = None
        self._matcher: re.Pattern[str] | None = None
        self._by_name: dict[str, DoseLimit] = {}

    # ---------------------------------------------------------------- loading

    def _load(self) -> list[DoseLimit]:
        if self._loaded is not None:
            return self._loaded
        limits = [
            item if isinstance(item, DoseLimit) else DoseLimit.model_validate(item)
            for item in self._inline
        ]
        if self._reference:
            try:
                rows = json.loads(Path(self._reference).read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                raise ConfigError(
                    f"Could not read the dose reference file {self._reference!r}.",
                    context={"reference": self._reference},
                ) from e
            if not isinstance(rows, list):
                raise ConfigError(
                    "The dose reference file must hold a JSON list of limits.",
                    context={"reference": self._reference},
                )
            limits.extend(DoseLimit.model_validate(row) for row in rows)
        self._by_name = {}
        for limit in limits:
            for name in (limit.drug, *limit.aliases):
                self._by_name[name.lower()] = limit
        if self._by_name:
            names = sorted(self._by_name, key=len, reverse=True)
            self._matcher = re.compile(
                r"(?<![A-Za-z0-9])(?:" + "|".join(re.escape(n) for n in names) + r")"
                r"(?![A-Za-z0-9])",
                re.IGNORECASE,
            )
        self._loaded = limits
        return limits

    def provenance(self) -> ComponentProvenance:
        """Records the number of limits and a hash of the table, not its content."""
        base = super().provenance()
        try:
            limits = self._load()
        except (ConfigError, ValueError):
            return base.model_copy(update={"config": {"reference": "unreadable"}})
        digest = hashlib.sha256(
            json.dumps(
                sorted((lim.model_dump() for lim in limits), key=lambda d: d["drug"]),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        return base.model_copy(
            update={"config": {"limits": len(limits), "reference_sha256": digest}}
        )

    # --------------------------------------------------------------- checking

    def check(self, payload: Any) -> GuardrailResult:
        if not isinstance(payload, str):
            return self._result(True, GuardrailSeverity.INFO, "Not a text payload.")
        try:
            limits = self._load()
        except (ConfigError, ValueError):
            return self._result(
                False,
                GuardrailSeverity.WARNING,
                "The dose reference could not be loaded, so no dose was checked.",
            )
        if not limits or self._matcher is None:
            return self._result(
                False,
                GuardrailSeverity.WARNING,
                "No dose reference is configured, so no dose was checked. Pass "
                "reference= or limits=.",
            )

        blocking: list[dict[str, JsonValue]] = []
        warnings: list[dict[str, JsonValue]] = []
        spans: list[TextSpan] = []
        checked = 0
        mentions = list(self._matcher.finditer(payload))
        for i, mention in enumerate(mentions):
            limit = self._by_name[mention.group(0).lower()]
            # A statement is read only between the neighbouring drug mentions, so
            # "a and b 50 mg" does not judge a against b's dose.
            lower = mentions[i - 1].end() if i > 0 else 0
            upper = mentions[i + 1].start() if i + 1 < len(mentions) else len(payload)
            found = self._statement(payload, mention, lower, upper)
            if found is None:
                continue
            checked += 1
            for severity, problem in self._problems(limit, *found):
                (blocking if severity is GuardrailSeverity.BLOCK else warnings).append(
                    problem
                )
                spans.append(
                    TextSpan(
                        start=mention.start(),
                        end=mention.end(),
                        label="dose",
                        confidence=1.0,
                    )
                )

        if blocking or warnings:
            severity = (
                GuardrailSeverity.BLOCK if blocking else GuardrailSeverity.WARNING
            )
            return GuardrailResult(
                passed=False,
                severity=severity,
                guardrail_key=self.registry_key,
                message=(
                    f"{len(blocking) + len(warnings)} dose statement problem(s) "
                    f"against the reference ({len(blocking)} over a limit)."
                ),
                spans=spans[:50],
                details={
                    "problems": [*blocking, *warnings][:20],
                    "checked": checked,
                },
            )
        if checked == 0:
            return self._result(
                True,
                GuardrailSeverity.INFO,
                "No readable dose statement for a drug in the reference was found; "
                "drugs outside the reference are not checked.",
            )
        return self._result(
            True,
            GuardrailSeverity.INFO,
            f"{checked} dose statement(s) within the reference limits.",
        )

    def _result(
        self, passed: bool, severity: GuardrailSeverity, message: str
    ) -> GuardrailResult:
        return GuardrailResult(
            passed=passed,
            severity=severity,
            guardrail_key=self.registry_key,
            message=message,
        )

    @staticmethod
    def _statement(
        text: str, mention: re.Match[str], lower: int, upper: int
    ) -> tuple[re.Match[str], str | None, int | None] | None:
        """The dose, route and per-day count in the clause around a drug mention.

        The dose is looked for after the drug name first ("metformin 500 mg") and then
        before it ("give 500 mg of metformin"), within the clause and between
        ``lower`` and ``upper`` (the neighbouring drug mentions).
        """
        floor = max(lower, mention.start() - _WINDOW)
        clause_start = floor
        for m in _CLAUSE_END.finditer(text, floor, mention.start()):
            clause_start = m.end()
        ceiling = min(upper, mention.end() + _WINDOW, len(text))
        end_match = _CLAUSE_END.search(text, mention.end(), ceiling)
        clause_end = end_match.start() if end_match else ceiling

        after = text[mention.end() : clause_end]
        before = text[clause_start : mention.start()]
        dose = _DOSE.search(after)
        if dose is None:
            candidates = list(_DOSE.finditer(before))
            dose = candidates[-1] if candidates else None
        if dose is None:
            return None
        clause = text[clause_start:clause_end]
        route_match = _ROUTE.search(clause)
        route = (
            _ROUTE_CANONICAL.get(route_match.group("route").lower())
            if route_match
            else None
        )
        freq_match = _FREQUENCY.search(clause)
        return dose, route, _per_day(freq_match) if freq_match else None

    @staticmethod
    def _problems(
        limit: DoseLimit, dose: re.Match[str], route: str | None, per_day: int | None
    ) -> list[tuple[GuardrailSeverity, dict[str, JsonValue]]]:
        problems: list[tuple[GuardrailSeverity, dict[str, JsonValue]]] = []
        unit = dose.group("unit")
        low = _to_unit(float(dose.group("low")), unit, limit.unit)
        high_text = dose.group("high")
        high = _to_unit(float(high_text), unit, limit.unit) if high_text else low
        if low is None or high is None:
            problems.append(
                (
                    GuardrailSeverity.WARNING,
                    {
                        "drug": limit.drug,
                        "problem": "unit_mismatch",
                        "stated_unit": unit,
                        "reference_unit": limit.unit,
                    },
                )
            )
            return problems
        top = max(low, high)
        bottom = min(low, high)
        if limit.max_single is not None and top > limit.max_single:
            problems.append(
                (
                    GuardrailSeverity.BLOCK,
                    {
                        "drug": limit.drug,
                        "problem": "over_max_single",
                        "value": top,
                        "limit": limit.max_single,
                        "unit": limit.unit,
                        "source": limit.source,
                    },
                )
            )
        if limit.min_single is not None and bottom < limit.min_single:
            problems.append(
                (
                    GuardrailSeverity.WARNING,
                    {
                        "drug": limit.drug,
                        "problem": "under_min_single",
                        "value": bottom,
                        "limit": limit.min_single,
                        "unit": limit.unit,
                        "source": limit.source,
                    },
                )
            )
        if limit.max_daily is not None and per_day is not None:
            total = top * per_day
            if total > limit.max_daily:
                problems.append(
                    (
                        GuardrailSeverity.BLOCK,
                        {
                            "drug": limit.drug,
                            "problem": "over_max_daily",
                            "value": total,
                            "limit": limit.max_daily,
                            "unit": limit.unit,
                            "doses_per_day": per_day,
                            "source": limit.source,
                        },
                    )
                )
        if limit.routes and route is not None and route not in limit.routes:
            problems.append(
                (
                    GuardrailSeverity.WARNING,
                    {
                        "drug": limit.drug,
                        "problem": "route_not_listed",
                        "route": route,
                        "allowed": list(limit.routes),
                        "source": limit.source,
                    },
                )
            )
        return problems
