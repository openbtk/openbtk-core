"""
Synthetic EHR/FHIR fixture generator for testing.

Generates minimal valid FHIR R4 resources with fake data. For full-scale
synthetic EHR generation, use the Synthea project (Java-based, generates
complete synthetic patient populations) — this module provides lightweight
Python-only fixtures for unit testing.
"""
from __future__ import annotations

import uuid
from datetime import date, timedelta

from faker import Faker

_fake = Faker()
Faker.seed(42)


def generate_fhir_patient() -> dict:
    """Generate a minimal synthetic FHIR R4 Patient resource."""
    return {
        "resourceType": "Patient",
        "id": str(uuid.uuid4()),
        "gender": _fake.random_element(["male", "female"]),
        "birthDate": _fake.date_of_birth(minimum_age=18, maximum_age=90).isoformat(),
    }


def generate_fhir_condition(patient_id: str, snomed_code: str = "44054006") -> dict:
    """Generate a minimal synthetic FHIR R4 Condition resource.

    Default SNOMED code 44054006 = "Diabetes mellitus type 2" (example only).
    """
    return {
        "resourceType": "Condition",
        "id": str(uuid.uuid4()),
        "subject": {"reference": f"Patient/{patient_id}"},
        "code": {
            "coding": [{
                "system": "http://snomed.info/sct",
                "code": snomed_code,
                "display": "Type 2 diabetes mellitus",
            }]
        },
        "onsetDateTime": (date.today() - timedelta(days=365)).isoformat(),
    }


def generate_fhir_observation(patient_id: str, loinc_code: str = "2093-3") -> dict:
    """Generate a minimal synthetic FHIR R4 Observation (lab result) resource.

    Default LOINC code 2093-3 = "Cholesterol [Mass/volume] in Serum" (example).
    """
    return {
        "resourceType": "Observation",
        "id": str(uuid.uuid4()),
        "status": "final",
        "subject": {"reference": f"Patient/{patient_id}"},
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": loinc_code,
                "display": "Cholesterol [Mass/volume] in Serum",
            }]
        },
        "valueQuantity": {
            "value": round(_fake.random.uniform(120, 240), 1),
            "unit": "mg/dL",
        },
        "effectiveDateTime": date.today().isoformat(),
    }


def generate_fhir_medication_request(patient_id: str, rxnorm_code: str = "1049502") -> dict:
    """Generate a minimal synthetic FHIR R4 MedicationRequest resource.

    Default RxNorm code 1049502 = "Metformin 500 MG Oral Tablet" (example).
    """
    return {
        "resourceType": "MedicationRequest",
        "id": str(uuid.uuid4()),
        "status": "active",
        "subject": {"reference": f"Patient/{patient_id}"},
        "medicationCodeableConcept": {
            "coding": [{
                "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                "code": rxnorm_code,
                "display": "Metformin 500 MG Oral Tablet",
            }]
        },
    }


def generate_fhir_bundle(n_conditions: int = 2, n_observations: int = 3) -> dict:
    """Generate a complete synthetic FHIR Bundle for one patient.

    Returns:
        A FHIR Bundle resource containing one Patient and related resources.
    """
    patient = generate_fhir_patient()
    patient_id = patient["id"]

    entries = [{"resource": patient}]
    entries += [{"resource": generate_fhir_condition(patient_id)} for _ in range(n_conditions)]
    entries += [{"resource": generate_fhir_observation(patient_id)} for _ in range(n_observations)]
    entries.append({"resource": generate_fhir_medication_request(patient_id)})

    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": entries,
    }
