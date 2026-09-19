"""Integration test (M7): guardrails composed via ``GuardrailPipeline``,
run against REAL cross-milestone components -- not test doubles.

Two real scenarios, not proven by any single unit test:

1. **EHR guardrails against a real FHIR-loaded ``PatientRecord``.** The
   same Synthea-shaped bundle tests/integration/test_ehr_pipeline.py (M6
   task 6.7) builds, loaded through the real ``FHIRLoader``, then checked
   by ``EHRCodeValidityGuardrail``/``ReferentialIntegrityGuardrail``/
   ``UnitPlausibilityGuardrail`` together via one ``GuardrailPipeline``
   run -- proving they compose over one real payload, not just that each
   passes its own contract test in isolation.
2. **PHI leakage across the M6 cross-modal seam.** A real
   ``PatientTimelineSerializer`` output (which embeds a deliberately
   PHI-shaped value the same way M6's own integration test does) is
   checked by ``PHILeakageGuardrail`` before and after the real
   ``DeidEngine`` runs on it -- proving the guardrail actually detects
   real PHI and actually stops detecting it once real de-identification
   has run, not merely that ``check()`` returns a well-typed result.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from openbtk.data import ehr as _ehr  # noqa: F401 -- registers FHIRLoader
from openbtk.deid import DeidEngine, DeidMode
from openbtk.guardrails.ehr import (
    EHRCodeValidityGuardrail,
    ReferentialIntegrityGuardrail,
    UnitPlausibilityGuardrail,
)
from openbtk.guardrails.phi_leakage import PHILeakageGuardrail
from openbtk.guardrails.pipeline import GuardrailPipeline
from openbtk.pipelines import PatientTimelineSerializer

if TYPE_CHECKING:
    from pathlib import Path

_SSN_VALUE = "123-45-6789"  # phi-fixture-ok: synthetic, unassigned test value


def _synthea_shaped_bundle(
    patient_id: str, *, with_referential_violation: bool
) -> dict[str, Any]:
    condition_display = f"Community-acquired pneumonia (contact SSN {_SSN_VALUE})"
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
                                "code": "385093006",
                                "display": condition_display,
                            }
                        ]
                    },
                    "onsetDateTime": (
                        "2030-01-01T08:10:00Z"
                        if with_referential_violation
                        else "2024-03-14T08:10:00Z"
                    ),
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
                                "code": "6690-2",
                                "display": "WBC",
                            }
                        ]
                    },
                    "valueQuantity": {"value": 7.5, "unit": "10*3/uL"},
                    "effectiveDateTime": "2024-03-14T08:20:00Z",
                }
            },
        ],
    }


def _load_patient(directory: Path, *, with_referential_violation: bool = False) -> Any:
    from openbtk.data.ehr.fhir import FHIRLoader

    bundle = _synthea_shaped_bundle(
        "pt-1", with_referential_violation=with_referential_violation
    )
    (directory / "pt-1.json").write_text(json.dumps(bundle), encoding="utf-8")
    return next(FHIRLoader().load(str(directory)))


def _code_validity_guardrail_for_this_bundle(
    directory: Path,
) -> EHRCodeValidityGuardrail:
    """The default ``EHRCodeValidityGuardrail()`` only knows ICD-10-CM
    (``BundledMinimalBackend``'s own real, disclosed scope) -- this
    bundle's real SNOMED/LOINC codes (Synthea's actual convention, the
    same ones tests/integration/test_ehr_pipeline.py uses) need a richer
    backend injected, the same way a real deployment would. A
    ``LocalVocabBackend`` covering exactly this test's two codes is that
    real backend, not a shortcut around the guardrail's own logic."""
    path = directory / "vocab.csv"
    path.write_text(
        "code,system,display\n"
        "385093006,SNOMED,Community-acquired pneumonia\n"
        "6690-2,LOINC,WBC\n",
        encoding="utf-8",
    )
    from openbtk.terminology.local import LocalVocabBackend

    return EHRCodeValidityGuardrail(terminology=LocalVocabBackend(path=str(path)))


class TestEHRGuardrailsComposeOverARealPatientRecord:
    def test_a_well_formed_patient_record_passes_every_ehr_guardrail(
        self, tmp_path: Path
    ) -> None:
        patient = _load_patient(tmp_path)
        pipeline = GuardrailPipeline(
            [
                _code_validity_guardrail_for_this_bundle(tmp_path),
                ReferentialIntegrityGuardrail(),
                UnitPlausibilityGuardrail(),
            ],
            short_circuit=False,
        )
        result = pipeline.run(patient)
        assert result.passed is True
        assert len(result.results) == 3

    def test_default_code_validity_backend_honestly_cannot_verify_snomed_loinc(
        self, tmp_path: Path
    ) -> None:
        """The real, disclosed limitation of BundledMinimalBackend (ICD-10-CM
        only) actually fires for this bundle's real SNOMED/LOINC codes --
        proven, not just documented."""
        patient = _load_patient(tmp_path)
        result = EHRCodeValidityGuardrail().check(patient)
        assert result.passed is False
        assert "SNOMED:385093006" in result.details["invalid"]
        assert "LOINC:6690-2" in result.details["invalid"]

    def test_a_real_referential_violation_is_caught_and_attributed(
        self, tmp_path: Path
    ) -> None:
        patient = _load_patient(tmp_path, with_referential_violation=True)
        pipeline = GuardrailPipeline(
            [
                _code_validity_guardrail_for_this_bundle(tmp_path),
                ReferentialIntegrityGuardrail(),
                UnitPlausibilityGuardrail(),
            ],
            short_circuit=False,
        )
        result = pipeline.run(patient)
        assert result.passed is False
        referential_result = next(
            r for r in result.results if r.guardrail_key == "guardrail.ehr.referential"
        )
        assert referential_result.passed is False
        # The other two guardrails are unaffected by this specific violation.
        code_validity_result = next(
            r
            for r in result.results
            if r.guardrail_key == "guardrail.ehr.code_validity"
        )
        assert code_validity_result.passed is True


class TestPHILeakageAcrossTheCrossModalSeam:
    def test_phi_is_detected_before_deid_and_gone_after(self, tmp_path: Path) -> None:
        patient = _load_patient(tmp_path)
        text_record = PatientTimelineSerializer().serialize(patient)
        assert _SSN_VALUE in text_record.text  # sanity: really embedded

        guardrail = PHILeakageGuardrail()
        before = guardrail.check(text_record.text)
        assert before.passed is False

        engine = DeidEngine(mode=DeidMode.REDACT)
        deid_result = engine.deidentify(text_record.text, patient_id=patient.patient_id)
        assert _SSN_VALUE not in deid_result.text

        after = guardrail.check(deid_result.text)
        assert after.passed is True
        # The real clinical content survives de-identification.
        assert "pneumonia" in deid_result.text.lower()
