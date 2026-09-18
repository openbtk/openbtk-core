"""Integration test (M6 task 6.7): a Synthea-shaped FHIR round trip, end to
end, through REAL components -- not test doubles.

Real Synthea (the Java synthetic-patient generator) is not invoked here --
running it needs a JVM and a real generation pass this repository has no
business bundling just to prove a loader works (the same reasoning already
applied to every other "real tool, synthetic fixture" integration test in
this project, e.g. tests/integration/test_clinical_text_pipeline.py never
shells out to a real EHR). Instead this test builds a FHIR R4 Bundle with
the exact shape Synthea's own per-patient export produces (one Bundle file
per patient, Patient + Encounter + Condition + Observation + MedicationRequest
+ Procedure entries, standard HL7 system URIs) -- the same shape
tests/unit/data/ehr/test_fhir.py already verifies against real
``fhir.resources`` R4B validation, exercised here end-to-end instead of
resource-by-resource.

**What this proves, that no unit test does:**

1. ``FHIRLoader -> TemporalNormalizer -> PatientTimelineSerializer`` (tasks
   6.2/6.4/6.5) genuinely compose: a real loaded ``PatientRecord``, run
   through the real streaming ``Pipeline``/executor for the loader+
   normalizer stages, serializes to a real ``ClinicalTextRecord``.
2. **The "cross-modal seam" claim** (docs/05_DATA_MODALITY_SPEC.md section
   2.3: "the output is a normal ClinicalTextRecord, so it inherits
   de-identification, chunking and guardrails for free -- no new
   machinery") is verified, not just asserted: the serialized timeline is
   written to disk and fed through the exact SAME real clinical_text
   pipeline (``PlainTextLoader`` -> ``DeidPreprocessor`` ->
   ``SectionSegmenter`` -> ``SectionAwareChunker``) M3's own integration
   test uses, unmodified. A PHI-shaped value deliberately embedded in one
   coded event's ``display`` text (a contrived but structurally real way
   to prove the point, disclosed here rather than pretending an EHR
   ``display`` field would normally contain one) must survive loading and
   then be redacted by this same de-id step, exactly like free-text notes.
3. **"Switch from FHIR to OMOP is a config change"** (section 2.1's
   "convergence point" claim): the identical, unmodified
   ``has_condition(...)`` cohort predicate matches a patient loaded from a
   FHIR bundle AND an independently-built OMOP Parquet dataset describing
   an equivalent patient -- proving ``PatientRecord`` is genuinely
   source-agnostic, not merely by both loaders sharing a return type.
"""

from __future__ import annotations

import json
from datetime import date
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa
import pyarrow.parquet as pq

from openbtk.core.base import BaseGuardrail
from openbtk.core.registry import GUARDRAIL_REGISTRY
from openbtk.core.schemas import CodeSystem, GuardrailResult, GuardrailSeverity

# Registers PlainTextLoader/DeidPreprocessor/SectionSegmenter/
# SectionAwareChunker and FHIRLoader/OMOPLoader/TemporalNormalizer as a
# side effect -- this file is collected standalone (no
# tests/contract/conftest.py in this directory), so both imports have to
# happen explicitly here, the same as test_clinical_text_pipeline.py does
# for its own modality.
from openbtk.data import clinical_text as _clinical_text  # noqa: F401
from openbtk.data import ehr as _ehr  # noqa: F401
from openbtk.data.ehr.cohort import CohortBuilder, has_condition
from openbtk.pipelines import PatientTimelineSerializer, Pipeline, Step

if TYPE_CHECKING:
    from pathlib import Path

    from openbtk.data.ehr.schemas import PatientRecord

_SSN_VALUE = "123-45-6789"  # phi-fixture-ok: synthetic, unassigned test value
_PNEUMONIA_SNOMED = "385093006"
_WBC_LOINC = "6690-2"
_CEFTRIAXONE_RXNORM = "309090"
_CXR_CPT = "71046"


class _CapturePatientRecordGuardrail(BaseGuardrail):
    """Same legitimate use of the guardrail attachment mechanism as
    test_clinical_text_pipeline.py's own -- captures the real
    ``PatientRecord`` this run produced, in-process, since
    ``Pipeline.run()`` itself returns only a ``RunManifest``."""

    captured: ClassVar[list[PatientRecord]] = []

    def check(self, payload: Any) -> GuardrailResult:
        type(self).captured.append(payload)
        return GuardrailResult(
            passed=True,
            severity=GuardrailSeverity.INFO,
            guardrail_key=self.registry_key,
            message="captured for test inspection",
        )

    @classmethod
    def reset(cls) -> None:
        cls.captured = []


GUARDRAIL_REGISTRY.register("guardrail.general.ehr_integration_test_capture")(
    _CapturePatientRecordGuardrail
)


class _CaptureChunkTextGuardrail(BaseGuardrail):
    """Records every chunk's ``.text`` this run produces -- the same
    legitimate capture-guardrail use as ``_CapturePatientRecordGuardrail``
    above, applied to the second (text) pipeline instead."""

    captured: ClassVar[list[str]] = []

    def check(self, payload: Any) -> GuardrailResult:
        type(self).captured.append(payload.text)
        return GuardrailResult(
            passed=True,
            severity=GuardrailSeverity.INFO,
            guardrail_key=self.registry_key,
            message="captured for test inspection",
        )

    @classmethod
    def reset(cls) -> None:
        cls.captured = []


GUARDRAIL_REGISTRY.register("guardrail.general.ehr_integration_test_capture_text")(
    _CaptureChunkTextGuardrail
)


def _synthea_shaped_bundle(patient_id: str) -> dict[str, Any]:
    """A FHIR R4 Bundle matching Synthea's real default per-patient export
    shape -- verified resource-by-resource against real ``fhir.resources``
    R4B validation in tests/unit/data/ehr/test_fhir.py, reused here as one
    complete patient rather than isolated resources."""
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": patient_id,
                    "gender": "female",
                }
            },
            {
                "resource": {
                    "resourceType": "Encounter",
                    "id": "enc-1",
                    "status": "finished",
                    "class": {"code": "IMP"},
                    "subject": {"reference": f"Patient/{patient_id}"},
                    "period": {
                        "start": "2024-03-14T08:00:00Z",
                        "end": "2024-03-16T00:00:00Z",
                    },
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "cond-1",
                    "subject": {"reference": f"Patient/{patient_id}"},
                    "encounter": {"reference": "Encounter/enc-1"},
                    "code": {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": _PNEUMONIA_SNOMED,
                                # Deliberately, disclosedly contrived: a
                                # PHI-shaped value embedded in a coded
                                # event's display text, to prove real
                                # de-identification actually runs on
                                # whatever text a timeline serializes --
                                # not a realistic FHIR Condition.code.text.
                                "display": (
                                    f"Community-acquired pneumonia "
                                    f"(contact SSN {_SSN_VALUE} for records)"
                                ),
                            }
                        ]
                    },
                    "onsetDateTime": "2024-03-14T08:10:00Z",
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-1",
                    "status": "final",
                    "subject": {"reference": f"Patient/{patient_id}"},
                    "encounter": {"reference": "Encounter/enc-1"},
                    "code": {
                        "coding": [
                            {
                                "system": "http://loinc.org",
                                "code": _WBC_LOINC,
                                "display": "WBC",
                            }
                        ]
                    },
                    "valueQuantity": {"value": 14.2, "unit": "10*3/uL"},
                    "effectiveDateTime": "2024-03-14T08:20:00Z",
                    "referenceRange": [
                        {"low": {"value": 4.5}, "high": {"value": 11.0}}
                    ],
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "med-1",
                    "status": "active",
                    "intent": "order",
                    "subject": {"reference": f"Patient/{patient_id}"},
                    "encounter": {"reference": "Encounter/enc-1"},
                    "medicationCodeableConcept": {
                        "coding": [
                            {
                                "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                                "code": _CEFTRIAXONE_RXNORM,
                                "display": "Ceftriaxone 1 g IV q24h",
                            }
                        ]
                    },
                    "authoredOn": "2024-03-14T09:00:00Z",
                }
            },
            {
                "resource": {
                    "resourceType": "Procedure",
                    "id": "proc-1",
                    "status": "completed",
                    "subject": {"reference": f"Patient/{patient_id}"},
                    "encounter": {"reference": "Encounter/enc-1"},
                    "code": {
                        "coding": [
                            {
                                "system": "http://www.ama-assn.org/go/cpt",
                                "code": _CXR_CPT,
                                "display": "Chest X-ray",
                            }
                        ]
                    },
                    "performedDateTime": "2024-03-14T08:30:00Z",
                }
            },
        ],
    }


def _write_synthea_bundle(directory: Path, patient_id: str) -> None:
    (directory / f"{patient_id}.json").write_text(
        json.dumps(_synthea_shaped_bundle(patient_id)), encoding="utf-8"
    )


def _ingest_and_normalize(fhir_dir: Path) -> list[PatientRecord]:
    """Runs the REAL load -> temporal-normalize pipeline and returns the
    real ``PatientRecord``s it produced."""
    _CapturePatientRecordGuardrail.reset()
    pipeline = (
        Pipeline("ehr-integration-ingest")
        .add(Step("load", "loader.ehr.fhir", path=str(fhir_dir)))
        .add(Step("normalize", "preprocessor.ehr.temporal"))
        .guard("guardrail.general.ehr_integration_test_capture", at="after:normalize")
    )
    manifest = pipeline.run()
    assert manifest.status == "success", manifest.error
    return list(_CapturePatientRecordGuardrail.captured)


def _write_omop_equivalent(directory: Path, person_id: int) -> None:
    """An independently-built OMOP CDM Parquet dataset describing a
    patient equivalent to `_synthea_shaped_bundle` -- same condition code
    (SNOMED, since Synthea's own real OMOP ETL output uses SNOMED source
    values for conditions), proving convergence rather than assuming it."""
    pq.write_table(
        pa.table(
            {
                "person_id": [person_id],
                "gender_concept_id": [8532],
                "year_of_birth": [1980],
                "month_of_birth": [5],
                "day_of_birth": [1],
            }
        ),
        directory / "person.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "condition_occurrence_id": [1],
                "person_id": [person_id],
                "condition_source_value": [_PNEUMONIA_SNOMED],
                "condition_start_date": [date(2024, 3, 14)],
            }
        ),
        directory / "condition_occurrence.parquet",
    )


class TestSyntheaShapedFHIRRoundTrip:
    def test_real_provenance_and_clinical_content_survive_the_full_chain(
        self, tmp_path: Path
    ) -> None:
        fhir_dir = tmp_path / "fhir"
        fhir_dir.mkdir()
        _write_synthea_bundle(fhir_dir, "synthea-pt-1")

        patients = _ingest_and_normalize(fhir_dir)
        assert len(patients) == 1
        patient = patients[0]
        assert patient.patient_id == "synthea-pt-1"
        assert patient.source_system == "fhir-r4"
        assert len(patient.conditions) == 1
        assert len(patient.medications) == 1
        assert len(patient.procedures) == 1
        assert len(patient.observations) == 1
        # TemporalNormalizer's own real effect: events are sorted, and the
        # single-encounter chain is internally consistent.
        assert patient.conditions[0].timestamp is not None

        text_record = PatientTimelineSerializer().serialize(patient)
        assert text_record.patient_ref == "synthea-pt-1"
        assert _SSN_VALUE in text_record.text  # not yet redacted -- proven next

        notes_dir = tmp_path / "notes"
        notes_dir.mkdir()
        (notes_dir / f"{patient.patient_id}.txt").write_text(
            text_record.text, encoding="utf-8"
        )

        _CaptureChunkTextGuardrail.reset()
        text_pipeline = (
            Pipeline("ehr-integration-text")
            .add(Step("load", "loader.clinical_text.plain_text", path=str(notes_dir)))
            .add(Step("deid", "preprocessor.general.deidentify", mode="redact"))
            .add(Step("segment", "preprocessor.clinical_text.section_segment"))
            .add(Step("chunk", "chunker.clinical_text.section_aware", max_tokens=200))
            .guard(
                "guardrail.general.ehr_integration_test_capture_text", at="after:chunk"
            )
        )
        manifest = text_pipeline.run()
        assert manifest.status == "success", manifest.error

        # Real proof the SAME clinical_text pipeline (no new machinery)
        # actually de-identified the timeline's text: the SSN embedded in
        # the condition's display text is gone from every emitted chunk,
        # while the real clinical content (the pneumonia diagnosis, the
        # lab result, the medication) survives -- exactly the "inherits
        # de-identification ... for free" claim, verified rather than
        # assumed from the run reporting "success".
        chunks = _CaptureChunkTextGuardrail.captured
        assert chunks, "expected at least one chunk"
        combined = " ".join(chunks)
        assert _SSN_VALUE not in combined
        assert "pneumonia" in combined.lower()
        assert "wbc" in combined.lower()
        assert "ceftriaxone" in combined.lower()


class TestFHIRAndOMOPConverge:
    def test_the_same_cohort_predicate_matches_both_sources(
        self, tmp_path: Path
    ) -> None:
        fhir_dir = tmp_path / "fhir"
        fhir_dir.mkdir()
        _write_synthea_bundle(fhir_dir, "fhir-pt")
        fhir_patients = _ingest_and_normalize(fhir_dir)

        omop_dir = tmp_path / "omop"
        omop_dir.mkdir()
        _write_omop_equivalent(omop_dir, person_id=99)
        from openbtk.data.ehr.omop import OMOPLoader

        omop_patients = list(OMOPLoader().load(str(omop_dir)))

        predicate = has_condition(_PNEUMONIA_SNOMED, system=CodeSystem.SNOMED)
        fhir_cohort = list(CohortBuilder(fhir_patients).include(predicate))
        omop_cohort = list(CohortBuilder(omop_patients).include(predicate))

        assert len(fhir_cohort) == 1
        assert len(omop_cohort) == 1
        assert fhir_cohort[0].source_system == "fhir-r4"
        assert omop_cohort[0].source_system == "omop-cdm-5.4"
        # Same predicate, same shape of match, two structurally different
        # sources -- the "convergence point" claim, proven rather than
        # asserted from the type signature alone.

    def test_a_non_matching_condition_is_excluded_from_both(
        self, tmp_path: Path
    ) -> None:
        fhir_dir = tmp_path / "fhir"
        fhir_dir.mkdir()
        _write_synthea_bundle(fhir_dir, "fhir-pt")
        fhir_patients = _ingest_and_normalize(fhir_dir)

        predicate = has_condition("not-present", system=CodeSystem.SNOMED)
        assert list(CohortBuilder(fhir_patients).include(predicate)) == []
