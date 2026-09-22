"""``HL7v2Loader``: HL7 v2 message ingestion, streaming (FR-E-03,
docs/05_DATA_MODALITY_SPEC.md section 2.2). Wraps ``hl7apy`` for parsing.

HL7 v2 is a *message* format, not a *record* format: ``ADT^A01`` says a patient was
admitted, ``ORU^R01`` carries results, ``RDE`` an order. This loader folds the messages
about one patient into one :class:`~openbtk.data.ehr.schemas.PatientRecord`.

**Scope, disclosed rather than silently assumed:**

* A source is a directory of ``*.hl7`` files (or one file). A file holds one or more
  messages, separated the usual way (a new ``MSH`` segment; MLLP framing bytes are
  tolerated). Messages in a file are grouped by patient (``PID-3``); the same patient in
  two files is two records. Memory is one file, never the directory.
* Segments read: ``PID`` (id, birth date, sex, race, ethnic group, death), ``PV1``
  (visit number, class, admit and discharge times), ``DG1`` (conditions), ``OBX``
  (numeric and text observations), ``RXE`` and ``RXA`` (medications), ``PR1``
  (procedures). Anything else is ignored.
* **Names, addresses and phone numbers are never read**, though ``PID`` carries them: a
  record holds only what the model needs, so they cannot leak from it.
  ``patient_id`` is ``PID-3``, an identifier that is PHI; de-identify records before
  they leave your environment.
* A coded field is mapped to a :class:`~openbtk.core.schemas.CodeSystem` only through
  the coding-system names in HL7 table 0396 that mean the same thing (checked against
  HL7's published table, not recalled): ``SCT`` (SNOMED CT), ``LN`` (LOINC), ``I10C``
  (ICD-10-CM), ``RXNORM`` and ``C4`` (CPT-4). Plain ``I10`` is the WHO ICD-10, a
  different code set from ICD-10-CM, so it is **not** mapped. The alternate coding
  (components 4-6) is tried when the primary is not mappable. An event none of whose
  codings is mappable is skipped and counted in a ``loader.hl7v2.skipped`` log line,
  never raised: one unknown code should not lose a patient.
* A timestamp with no UTC offset is taken to be at ``default_utc_offset_hours``
  (default UTC). HL7 v2 leaves local time unlabelled, so this is a stated assumption;
  set it if your interface engine emits local time.
* Results marked deleted or not performed (``OBX-11`` of ``D`` or ``X``) are skipped.
  A repeated identical event (an admit followed by an update) is kept once.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

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
    from collections.abc import Iterator
    from typing import Any

log = get_logger(__name__)

_SOURCE_SYSTEM = "hl7v2"

# HL7 table 0396 names that denote exactly one CodeSystem OpenBTK models.
_SYSTEMS: dict[str, CodeSystem] = {
    "SCT": CodeSystem.SNOMED,
    "LN": CodeSystem.LOINC,
    "I10C": CodeSystem.ICD10CM,
    "RXNORM": CodeSystem.RXNORM,
    "C4": CodeSystem.CPT,
}

# HL7 table 0001 (administrative sex) onto the four values Demographics allows.
_SEX = {
    "M": "male",
    "F": "female",
    "O": "other",
    "A": "other",
    "X": "other",
    "U": "unknown",
    "N": "unknown",
}

# HL7 table 0004 (patient class).
_PATIENT_CLASS = {
    "I": "inpatient",
    "O": "outpatient",
    "E": "emergency",
    "P": "preadmit",
    "R": "recurring",
    "B": "obstetrics",
    "C": "commercial account",
    "N": "not applicable",
    "U": "unknown",
}

_TS = re.compile(
    r"^(?P<y>\d{4})(?P<mo>\d{2})?(?P<d>\d{2})?(?P<h>\d{2})?(?P<mi>\d{2})?(?P<s>\d{2})?"
    r"(?:\.\d{1,6})?(?P<tz>[+-]\d{4})?$"
)
_RANGE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*$")
_ESCAPES = {"F": "|", "S": "^", "T": "&", "R": "~", "E": "\\"}
_MLLP = "\x0b\x1c"


def _unescape(value: str) -> str:
    """Resolve the five standard HL7 escape sequences (``\\F\\`` and so on)."""
    if "\\" not in value:
        return value
    return re.sub(r"\\([FSTRE])\\", lambda m: _ESCAPES[m.group(1)], value)


def _parts(value: str | None, separator: str = "^") -> list[str]:
    return value.split(separator) if value else []


def _component(value: str | None, index: int) -> str | None:
    """Component ``index`` (1-based) of a field, unescaped, or ``None`` if empty."""
    parts = _parts(value)
    if index > len(parts):
        return None
    text = _unescape(parts[index - 1]).strip()
    return text or None


def _timestamp(value: str | None, offset_hours: float) -> datetime | None:
    if not value:
        return None
    match = _TS.match(value.split("^")[0].strip())
    if match is None:
        return None
    zone = timezone(timedelta(hours=offset_hours)) if offset_hours else UTC
    if match["tz"]:
        sign = 1 if match["tz"][0] == "+" else -1
        zone = timezone(
            sign * timedelta(hours=int(match["tz"][1:3]), minutes=int(match["tz"][3:5]))
        )
    try:
        return datetime(
            int(match["y"]),
            int(match["mo"] or 1),
            int(match["d"] or 1),
            int(match["h"] or 0),
            int(match["mi"] or 0),
            int(match["s"] or 0),
            tzinfo=zone,
        )
    except ValueError:
        return None


def _date(value: str | None) -> date | None:
    if not value:
        return None
    match = _TS.match(value.split("^")[0].strip())
    if match is None:
        return None
    try:
        return date(int(match["y"]), int(match["mo"] or 1), int(match["d"] or 1))
    except ValueError:
        return None


def _coding(value: str | None) -> tuple[str, CodeSystem, str | None] | None:
    """The first mappable ``(code, system, display)`` of a coded field, or ``None``."""
    for code_at in (1, 4):
        code = _component(value, code_at)
        system = _component(value, code_at + 2)
        if code and system and system.upper() in _SYSTEMS:
            return code, _SYSTEMS[system.upper()], _component(value, code_at + 1)
    return None


class _Patient:
    """What is known about one patient so far, while a file's messages are folded in."""

    def __init__(self, patient_id: str) -> None:
        self.patient_id = patient_id
        self.birth_date: date | None = None
        self.gender: str | None = None
        self.race: str | None = None
        self.ethnicity: str | None = None
        self.deceased = False
        self.deceased_date: date | None = None
        self.encounters: dict[str, Encounter] = {}
        self.conditions: dict[tuple[object, ...], CodedEvent] = {}
        self.medications: dict[tuple[object, ...], CodedEvent] = {}
        self.procedures: dict[tuple[object, ...], CodedEvent] = {}
        self.observations: dict[tuple[object, ...], Measurement] = {}

    def record(self) -> PatientRecord:
        return PatientRecord(
            patient_id=self.patient_id,
            demographics=Demographics(
                birth_date=self.birth_date,
                gender=self.gender,
                race=self.race,
                ethnicity=self.ethnicity,
                deceased=self.deceased,
                deceased_date=self.deceased_date,
            ),
            encounters=list(self.encounters.values()),
            conditions=list(self.conditions.values()),
            medications=list(self.medications.values()),
            procedures=list(self.procedures.values()),
            observations=list(self.observations.values()),
            source_system=_SOURCE_SYSTEM,
        )


@LOADER_REGISTRY.register("loader.ehr.hl7v2")
class HL7v2Loader(BaseLoader[str, PatientRecord]):
    """Load ``PatientRecord``s from HL7 v2 messages. See the module docstring for the
    exact scope.

    Args:
        default_utc_offset_hours: The offset assumed for a timestamp that carries none.
        encoding: The text encoding of the files.

    Example:
        >>> import contextlib, io, tempfile
        >>> from pathlib import Path
        >>> message = "\\r".join([
        ...     "MSH|^~\\\\&|APP|FAC|RCV|FAC|20240314101500||ADT^A01|M1|P|2.5",
        ...     "PID|1||PT0001^^^HOSP^MR||DOE^JANE||19800601|F",
        ... ])
        >>> with tempfile.TemporaryDirectory() as d:
        ...     _ = (Path(d) / "a.hl7").write_text(message)
        ...     with contextlib.redirect_stdout(io.StringIO()):
        ...         records = list(HL7v2Loader().load(d))
        >>> (records[0].patient_id, records[0].demographics.gender)
        ('PT0001', 'female')
    """

    def __init__(
        self, *, default_utc_offset_hours: float = 0.0, encoding: str = "utf-8"
    ) -> None:
        self._offset = default_utc_offset_hours
        self._encoding = encoding

    def load(self, source: str) -> Iterator[PatientRecord]:
        """Raises:
        LoaderError: If ``source`` is neither a file nor a directory, a file cannot
            be read, or a message cannot be parsed. The message names the file and
            the message's position, never its content.
        """
        path = Path(source)
        if path.is_dir():
            files = sorted(p for p in path.iterdir() if p.suffix.lower() == ".hl7")
        elif path.is_file():
            files = [path]
        else:
            raise LoaderError(
                f"Not a file or directory: {source}",
                context={"modality": "ehr", "stage": "load"},
            )
        parser = require("hl7apy.parser", extra="ehr")
        consts = require("hl7apy.consts", extra="ehr")
        log.info("loader.start", modality="ehr", format="hl7v2", source=str(path))
        for file in files:
            yield from self._patients_in(file, parser, consts)

    # ------------------------------------------------------------------ files

    def _patients_in(
        self, file: Path, parser: Any, consts: Any
    ) -> Iterator[PatientRecord]:
        ctx = {"modality": "ehr", "stage": "load", "filename": file.name}
        try:
            text = file.read_text(encoding=self._encoding)
        except (OSError, UnicodeDecodeError) as e:
            raise LoaderError(f"Failed to read {file.name}.", context=ctx) from e
        patients: dict[str, _Patient] = {}
        skipped = 0
        for index, raw in enumerate(_split_messages(text), start=1):
            try:
                message = parser.parse_message(
                    raw,
                    validation_level=consts.VALIDATION_LEVEL.TOLERANT,
                    find_groups=False,
                )
            except Exception as e:  # hl7apy raises its own hierarchy plus ValueError
                raise LoaderError(
                    f"{file.name}: message {index} is not valid HL7 v2.",
                    context={**ctx, "message": index},
                ) from e
            skipped += self._fold(message, patients)
        if skipped:
            log.warning("loader.hl7v2.skipped", filename=file.name, events=skipped)
        for patient in patients.values():
            yield patient.record()

    # --------------------------------------------------------------- messages

    def _fold(self, message: Any, patients: dict[str, _Patient]) -> int:
        """Fold a message into ``patients``; returns how many events were skipped."""
        segments = [(seg.name, _fields(seg)) for seg in message.children]
        msh = next((f for name, f in segments if name == "MSH"), {})
        pid = next((f for name, f in segments if name == "PID"), None)
        if pid is None:
            return 0
        patient_id = _patient_id(pid.get(3))
        if patient_id is None:
            return 0
        patient = patients.setdefault(patient_id, _Patient(patient_id))
        trigger = _component(msh.get(9), 2)
        message_time = _timestamp(msh.get(7), self._offset)
        self._demographics(patient, pid)

        visit: str | None = None
        skipped = 0
        for name, f in segments:
            if name == "PV1":
                visit = self._visit(patient, f, trigger)
            elif name == "DG1":
                skipped += self._coded(
                    patient.conditions,
                    f.get(3),
                    _timestamp(f.get(5), self._offset),
                    visit,
                )
            elif name == "PR1":
                skipped += self._coded(
                    patient.procedures,
                    f.get(3),
                    _timestamp(f.get(5), self._offset),
                    visit,
                )
            elif name == "RXE":
                skipped += self._coded(
                    patient.medications, f.get(2), message_time, visit
                )
            elif name == "RXA":
                skipped += self._coded(
                    patient.medications,
                    f.get(5),
                    _timestamp(f.get(3), self._offset) or message_time,
                    visit,
                )
            elif name == "OBX":
                skipped += self._observation(patient, f, message_time, visit)
        return skipped

    def _demographics(self, patient: _Patient, pid: dict[int, str]) -> None:
        patient.birth_date = _date(pid.get(7)) or patient.birth_date
        sex = _component(pid.get(8), 1)
        patient.gender = _SEX.get(sex.upper()) if sex else patient.gender
        patient.race = _label(pid.get(10)) or patient.race
        patient.ethnicity = _label(pid.get(22)) or patient.ethnicity
        died = _date(pid.get(29))
        if died is not None:
            patient.deceased_date = died
        if died is not None or (_component(pid.get(30), 1) or "").upper() == "Y":
            patient.deceased = True

    def _visit(
        self, patient: _Patient, f: dict[int, str], trigger: str | None
    ) -> str | None:
        visit = _component(f.get(19), 1)
        if visit is None:
            return None
        start = _timestamp(f.get(44), self._offset)
        end = _timestamp(f.get(45), self._offset)
        klass = (_component(f.get(2), 1) or "").upper()
        status = "finished" if end is not None or trigger == "A03" else None
        status = status or ("in-progress" if trigger == "A01" else None)
        previous = patient.encounters.get(visit)
        patient.encounters[visit] = Encounter(
            encounter_id=visit,
            encounter_type=_PATIENT_CLASS.get(klass)
            or (previous.encounter_type if previous else None),
            start=start or (previous.start if previous else None),
            end=end or (previous.end if previous else None),
            status=status or (previous.status if previous else None),
        )
        return visit

    def _coded(
        self,
        store: dict[tuple[object, ...], CodedEvent],
        field: str | None,
        when: datetime | None,
        visit: str | None,
    ) -> int:
        coding = _coding(field)
        if coding is None:
            return 1 if field else 0
        code, system, display = coding
        store.setdefault(
            (code, system, when, visit),
            CodedEvent(
                code=code,
                system=system,
                display=display,
                timestamp=when,
                encounter_ref=visit,
            ),
        )
        return 0

    def _observation(
        self,
        patient: _Patient,
        f: dict[int, str],
        message_time: datetime | None,
        visit: str | None,
    ) -> int:
        if (_component(f.get(11), 1) or "").upper() in {"D", "X"}:
            return 0
        coding = _coding(f.get(3))
        if coding is None:
            return 1
        code, system, display = coding
        value: float | str | None
        raw = _unescape(f.get(5, "")).strip()
        kind = (_component(f.get(2), 1) or "").upper()
        if kind == "NM":
            try:
                value = float(raw)
            except ValueError:
                return 1
        elif kind in {"ST", "TX", "FT"}:
            value = raw or None
        elif kind in {"CE", "CWE"}:
            value = _component(f.get(5), 2) or _component(f.get(5), 1)
        else:
            return 1
        if value is None:
            return 0
        unit_system = (_component(f.get(6), 3) or "UCUM").upper()
        unit = _component(f.get(6), 1) if unit_system == "UCUM" else None
        bounds = _RANGE.match(_unescape(f.get(7, "")))
        when = _timestamp(f.get(14), self._offset) or message_time
        patient.observations.setdefault(
            (code, system, when, visit, value),
            Measurement(
                code=code,
                system=system,
                display=display,
                value=value,
                unit=unit,
                reference_range=(float(bounds[1]), float(bounds[2]))
                if bounds
                else None,
                timestamp=when,
                encounter_ref=visit,
            ),
        )
        return 0


# ---------------------------------------------------------------- helpers


def _split_messages(text: str) -> Iterator[str]:
    """Split a file into messages at each ``MSH`` segment, normalising line ends."""
    for mark in _MLLP:
        text = text.replace(mark, "")
    text = text.replace("\r\n", "\r").replace("\n", "\r")
    current: list[str] = []
    for line in text.split("\r"):
        if not line.strip():
            continue
        if line.startswith("MSH") and current:
            yield "\r".join(current)
            current = []
        if line[:3] in {"FHS", "BHS", "BTS", "FTS"}:
            continue
        current.append(line)
    if current:
        yield "\r".join(current)


def _fields(segment: Any) -> dict[int, str]:
    """A segment's populated fields as ``{position: ER7 text}``."""
    fields: dict[int, str] = {}
    for field in segment.children:
        tail = field.name.rsplit("_", 1)[-1]
        if tail.isdigit():
            fields[int(tail)] = field.to_er7()
    return fields


def _patient_id(pid3: str | None) -> str | None:
    """The medical-record-number identifier of ``PID-3`` if one is typed ``MR``, else
    the first identifier."""
    repeats = _parts(pid3, "~")
    for repeat in repeats:
        if (_component(repeat, 5) or "").upper() == "MR":
            return _component(repeat, 1)
    return _component(repeats[0], 1) if repeats else None


def _label(field: str | None) -> str | None:
    """The display text of a coded field, falling back to its code."""
    return _component(field, 2) or _component(field, 1)
