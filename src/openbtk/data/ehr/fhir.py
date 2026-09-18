"""``FHIRLoader``: R4 Bundle ingestion, streaming (docs/05_DATA_MODALITY_SPEC.md
section 2.2). Wraps ``fhir.resources`` (R4B -- the R4 technical-correction
release; Synthea and most real R4 exports validate against it cleanly).

**Scope, disclosed rather than silently assumed** (the same discipline
``loaders.py``'s own module docstring applies to ``CDALoader``/
``HL7v2Loader``):

* One JSON file is one FHIR ``Bundle``, and one ``Bundle`` is one patient's
  full record -- Synthea's own default per-patient export shape, and the
  shape task 6.7's round-trip test exercises directly. A loose,
  one-resource-per-file export (some real-world bulk-export tooling
  produces this instead) is not read by this loader; resources for a
  single patient would then be scattered across arbitrarily many files
  with no guaranteed co-location, which is a materially different, harder
  streaming problem than this task's exit criterion covers.
* "Per-resource" (this task's own roadmap wording) means: within one
  bundle, every one of the six resource types ``PatientRecord`` needs
  (``Patient``, ``Encounter``, ``Condition``, ``MedicationRequest``/
  ``MedicationStatement``, ``Procedure``, ``Observation``) is recognised
  and mapped to its right place -- not that each lives in its own file.
* A ``CodeableConcept`` is mapped to a :class:`~openbtk.core.schemas.CodeSystem`
  only when its ``coding.system`` URI is one of the six this loader knows
  (SNOMED CT, LOINC, ICD-10-CM, RxNorm, CPT, UCUM -- the standard HL7
  system URIs, not fabricated). A coding in an unrecognised system is
  skipped with a logged warning rather than raising: a real bundle can
  carry codes OpenBTK has no mapping for yet, and failing the whole
  patient over one unrecognised code would be a worse outcome than
  dropping just that one event.
* Race/ethnicity are read from the US Core ``us-core-race``/``us-core-
  ethnicity`` extensions when present (Synthea populates both) -- the
  real, standard extension URLs, not invented ones. Absent both, they
  stay ``None`` rather than guessed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

log = get_logger(__name__)

_SOURCE_SYSTEM = "fhir-r4"

# Real, standard HL7 FHIR terminology system URIs -- not fabricated. Verified
# against the FHIR R4 specification and Synthea's own real output.
_FHIR_SYSTEM_TO_CODE_SYSTEM: dict[str, CodeSystem] = {
    "http://snomed.info/sct": CodeSystem.SNOMED,
    "http://loinc.org": CodeSystem.LOINC,
    "http://hl7.org/fhir/sid/icd-10-cm": CodeSystem.ICD10CM,
    "http://www.nlm.nih.gov/research/umls/rxnorm": CodeSystem.RXNORM,
    "http://www.ama-assn.org/go/cpt": CodeSystem.CPT,
    "http://unitsofmeasure.org": CodeSystem.UCUM,
}

_US_CORE_RACE_URL = "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race"
_US_CORE_ETHNICITY_URL = (
    "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity"
)

_KNOWN_GENDERS = frozenset({"male", "female", "other", "unknown"})


def _reference_id(reference: str | None) -> str | None:
    """Strip a FHIR reference down to its bare resource id.

    ``"Encounter/enc-1"`` -> ``"enc-1"``; ``"urn:uuid:abc"`` -> ``"abc"``;
    already-bare ids pass through unchanged. This is what lets
    ``CodedEvent.encounter_ref``/``Measurement.encounter_ref`` match
    ``Encounter.encounter_id`` regardless of which reference style the
    source bundle used.
    """
    if reference is None:
        return None
    if "/" in reference:
        return reference.rsplit("/", 1)[-1]
    return reference.removeprefix("urn:uuid:")


def _first_known_coding(concept: Any) -> tuple[str, CodeSystem, str | None] | None:
    """Return ``(code, system, display)`` for the first coding in
    ``concept`` whose system URI this loader recognises, or ``None``."""
    if concept is None or not concept.coding:
        return None
    for coding in concept.coding:
        system = _FHIR_SYSTEM_TO_CODE_SYSTEM.get(coding.system or "")
        if system is not None and coding.code:
            return coding.code, system, coding.display
    return None


def _text_from_us_core_extension(extensions: Any, url: str) -> str | None:
    for ext in extensions or []:
        if ext.url != url:
            continue
        for sub in ext.extension or []:
            if sub.url == "text" and sub.valueString:
                return str(sub.valueString)
        for sub in ext.extension or []:
            if sub.url == "ombCategory" and sub.valueCoding and sub.valueCoding.display:
                return str(sub.valueCoding.display)
    return None


def _demographics_from_patient(patient: Any) -> Demographics:
    deceased_date = None
    deceased = bool(getattr(patient, "deceasedBoolean", None))
    if patient.deceasedDateTime is not None:
        deceased = True
        deceased_date = patient.deceasedDateTime.date()
    gender = patient.gender if patient.gender in _KNOWN_GENDERS else None
    return Demographics(
        birth_date=patient.birthDate,
        gender=gender,
        race=_text_from_us_core_extension(patient.extension, _US_CORE_RACE_URL),
        ethnicity=_text_from_us_core_extension(
            patient.extension, _US_CORE_ETHNICITY_URL
        ),
        deceased=deceased,
        deceased_date=deceased_date,
    )


def _encounter_from_resource(enc: Any) -> Encounter | None:
    if not enc.id:
        return None
    encounter_type = enc.class_fhir.code if enc.class_fhir is not None else None
    period = enc.period
    return Encounter(
        encounter_id=enc.id,
        encounter_type=encounter_type,
        start=period.start if period is not None else None,
        end=period.end if period is not None else None,
        status=enc.status,
    )


def _status_from_codeable_concept(concept: Any) -> str | None:
    if concept is not None and concept.coding:
        return str(concept.coding[0].code)
    return None


def _condition_to_coded_event(cond: Any) -> CodedEvent | None:
    coding = _first_known_coding(cond.code)
    if coding is None:
        return None
    code, system, display = coding
    timestamp = cond.onsetDateTime
    if timestamp is None and cond.onsetPeriod is not None:
        timestamp = cond.onsetPeriod.start
    if timestamp is None:
        timestamp = cond.recordedDate
    return CodedEvent(
        code=code,
        system=system,
        display=display,
        timestamp=timestamp,
        encounter_ref=_reference_id(
            cond.encounter.reference if cond.encounter else None
        ),
        status=_status_from_codeable_concept(cond.clinicalStatus),
    )


def _medication_request_to_coded_event(mr: Any) -> CodedEvent | None:
    coding = _first_known_coding(mr.medicationCodeableConcept)
    if coding is None:
        return None
    code, system, display = coding
    return CodedEvent(
        code=code,
        system=system,
        display=display,
        timestamp=mr.authoredOn,
        encounter_ref=_reference_id(mr.encounter.reference if mr.encounter else None),
        status=mr.status,
    )


def _medication_statement_to_coded_event(ms: Any) -> CodedEvent | None:
    coding = _first_known_coding(ms.medicationCodeableConcept)
    if coding is None:
        return None
    code, system, display = coding
    timestamp = ms.effectiveDateTime
    if timestamp is None and ms.effectivePeriod is not None:
        timestamp = ms.effectivePeriod.start
    return CodedEvent(
        code=code,
        system=system,
        display=display,
        timestamp=timestamp,
        encounter_ref=_reference_id(ms.context.reference if ms.context else None),
        status=ms.status,
    )


def _procedure_to_coded_event(proc: Any) -> CodedEvent | None:
    coding = _first_known_coding(proc.code)
    if coding is None:
        return None
    code, system, display = coding
    timestamp = proc.performedDateTime
    if timestamp is None and proc.performedPeriod is not None:
        timestamp = proc.performedPeriod.start
    return CodedEvent(
        code=code,
        system=system,
        display=display,
        timestamp=timestamp,
        encounter_ref=_reference_id(
            proc.encounter.reference if proc.encounter else None
        ),
        status=proc.status,
    )


def _observation_to_measurement(obs: Any) -> Measurement | None:
    coding = _first_known_coding(obs.code)
    if coding is None:
        return None
    code, system, display = coding
    value: float | str | None = None
    unit: str | None = None
    if obs.valueQuantity is not None:
        value = obs.valueQuantity.value
        unit = obs.valueQuantity.unit
    elif obs.valueString is not None:
        value = obs.valueString
    elif obs.valueCodeableConcept is not None and obs.valueCodeableConcept.coding:
        value = obs.valueCodeableConcept.coding[0].display
    reference_range = None
    if obs.referenceRange:
        low = obs.referenceRange[0].low
        high = obs.referenceRange[0].high
        if low is not None and high is not None:
            reference_range = (float(low.value), float(high.value))
    timestamp = obs.effectiveDateTime
    if timestamp is None and obs.effectivePeriod is not None:
        timestamp = obs.effectivePeriod.start
    return Measurement(
        code=code,
        system=system,
        display=display,
        value=value,
        unit=unit,
        reference_range=reference_range,
        timestamp=timestamp,
        encounter_ref=_reference_id(obs.encounter.reference if obs.encounter else None),
    )


@LOADER_REGISTRY.register("loader.ehr.fhir")
class FHIRLoader(BaseLoader[str, PatientRecord]):
    """Load one ``PatientRecord`` per FHIR R4 ``Bundle`` JSON file in a
    directory. See this module's own docstring for the loader's exact,
    disclosed scope.

    Example:
        >>> import contextlib, io, json, tempfile
        >>> from pathlib import Path
        >>> bundle = {
        ...     "resourceType": "Bundle",
        ...     "type": "collection",
        ...     "entry": [{"resource": {
        ...         "resourceType": "Patient", "id": "pt-1", "gender": "female",
        ...     }}],
        ... }
        >>> with tempfile.TemporaryDirectory() as d:
        ...     _ = (Path(d) / "pt-1.json").write_text(json.dumps(bundle))
        ...     with contextlib.redirect_stdout(io.StringIO()):
        ...         records = list(FHIRLoader().load(d))
        >>> records[0].patient_id
        'pt-1'
    """

    def load(self, source: str) -> Iterator[PatientRecord]:
        """Raises:
        LoaderError: If ``source`` is not a directory, a file is not
            valid JSON, is not a FHIR ``Bundle``, or a bundle contains no
            ``Patient`` resource.
        """
        directory = Path(source)
        if not directory.is_dir():
            raise LoaderError(
                f"Not a directory: {source}",
                context={"modality": "ehr", "stage": "load"},
            )
        fhir_r4b = require("fhir.resources.R4B", extra="ehr")
        log.info("loader.start", modality="ehr", source=str(directory))
        for path in sorted(directory.glob("*.json")):
            yield self._record_from_bundle_file(path, fhir_r4b)

    def _record_from_bundle_file(self, path: Path, fhir_r4b: Any) -> PatientRecord:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            raise LoaderError(
                f"Failed to read/parse {path.name}.",
                context={"modality": "ehr", "stage": "load", "filename": path.name},
            ) from e
        if raw.get("resourceType") != "Bundle":
            raise LoaderError(
                f"{path.name} is not a FHIR Bundle (resourceType="
                f"{raw.get('resourceType')!r}).",
                context={"modality": "ehr", "stage": "load", "filename": path.name},
            )
        bundle_cls = fhir_r4b.get_fhir_model_class("Bundle")
        try:
            bundle = bundle_cls.model_validate(raw)
        except Exception as e:
            raise LoaderError(
                f"{path.name} is not a valid FHIR R4 Bundle.",
                context={"modality": "ehr", "stage": "load", "filename": path.name},
            ) from e
        return self._record_from_bundle(bundle, path.name)

    def _record_from_bundle(self, bundle: Any, filename: str) -> PatientRecord:
        patient = None
        encounters: list[Encounter] = []
        conditions: list[CodedEvent] = []
        medications: list[CodedEvent] = []
        procedures: list[CodedEvent] = []
        observations: list[Measurement] = []

        for entry in bundle.entry or []:
            resource = entry.resource
            if resource is None:
                continue
            kind = type(resource).__name__
            if kind == "Patient":
                patient = resource
            elif kind == "Encounter":
                encounter = _encounter_from_resource(resource)
                if encounter is not None:
                    encounters.append(encounter)
                else:
                    log.warning("loader.ehr.encounter_missing_id", filename=filename)
            elif kind == "Condition":
                event = _condition_to_coded_event(resource)
                self._append_or_warn(conditions, event, "condition", filename)
            elif kind == "MedicationRequest":
                event = _medication_request_to_coded_event(resource)
                self._append_or_warn(medications, event, "medication_request", filename)
            elif kind == "MedicationStatement":
                event = _medication_statement_to_coded_event(resource)
                self._append_or_warn(
                    medications, event, "medication_statement", filename
                )
            elif kind == "Procedure":
                event = _procedure_to_coded_event(resource)
                self._append_or_warn(procedures, event, "procedure", filename)
            elif kind == "Observation":
                measurement = _observation_to_measurement(resource)
                if measurement is not None:
                    observations.append(measurement)
                else:
                    log.warning(
                        "loader.ehr.unrecognised_code_system",
                        filename=filename,
                        resource="observation",
                    )

        if patient is None:
            raise LoaderError(
                f"{filename} contains no Patient resource.",
                context={"modality": "ehr", "stage": "load", "filename": filename},
            )
        return PatientRecord(
            patient_id=patient.id,
            demographics=_demographics_from_patient(patient),
            encounters=encounters,
            conditions=conditions,
            medications=medications,
            procedures=procedures,
            observations=observations,
            source_system=_SOURCE_SYSTEM,
            metadata={"bundle_file": filename},
        )

    @staticmethod
    def _append_or_warn(
        target: list[CodedEvent], event: CodedEvent | None, resource: str, filename: str
    ) -> None:
        if event is not None:
            target.append(event)
        else:
            log.warning(
                "loader.ehr.unrecognised_code_system",
                filename=filename,
                resource=resource,
            )
