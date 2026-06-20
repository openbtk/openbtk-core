"""
Synthetic clinical text generators for testing.

CRITICAL: These generate FAKE clinical notes using Faker + templates.
No real patient data is ever used. Safe to commit to git, safe for CI.
"""
from __future__ import annotations

import uuid

from faker import Faker

_fake = Faker()
Faker.seed(42)  # deterministic test fixtures


def generate_discharge_summary() -> str:
    """Generate a synthetic discharge summary with standard sections."""
    name = _fake.name()
    age = _fake.random_int(min=18, max=95)
    admit_date = _fake.date_this_year()
    discharge_date = _fake.date_this_year()
    condition = _fake.random_element([
        "community-acquired pneumonia", "congestive heart failure exacerbation",
        "type 2 diabetes mellitus with hyperglycemia", "acute appendicitis",
        "chronic obstructive pulmonary disease exacerbation",
    ])
    medication = _fake.random_element([
        "metformin 500mg twice daily", "lisinopril 10mg once daily",
        "albuterol inhaler as needed", "amoxicillin 500mg three times daily",
    ])

    return f"""Chief Complaint
Patient presents with worsening symptoms over the past 3 days.

History of Present Illness
{name} is a {age} year old patient admitted on {admit_date} with {condition}.
Patient reports progressive symptoms prior to admission. No prior similar
episodes reported.

Past Medical History
Hypertension, hyperlipidemia, type 2 diabetes mellitus.

Medications
{medication}.

Physical Exam
Vital signs stable on discharge. Lungs clear to auscultation bilaterally.
Cardiac exam regular rate and rhythm, no murmurs.

Assessment and Plan
Patient improved with treatment for {condition}. Discharged in stable
condition on {discharge_date}. Follow up with primary care in 1 week.

Discharge Instructions
Continue home medications as prescribed. Return to ED if symptoms worsen.
"""


def generate_history_and_physical() -> str:
    """Generate a synthetic H&P note."""
    age = _fake.random_int(min=18, max=95)
    complaint = _fake.random_element([
        "chest pain", "shortness of breath", "abdominal pain", "headache",
    ])

    return f"""Chief Complaint
{complaint.capitalize()} for 2 days.

History of Present Illness
{age} year old presenting with {complaint}. Onset was gradual. Associated
symptoms include mild fatigue. No fever reported.

Past Surgical History
Appendectomy in childhood.

Family History
Father with history of coronary artery disease. Mother with hypertension.

Social History
Non-smoker. Occasional alcohol use. Works as an office administrator.

Review of Systems
Negative for fever, chills, nausea, vomiting. Positive for {complaint}.

Physical Exam
Alert and oriented x3. Vital signs within normal limits.

Assessment
Likely musculoskeletal etiology versus cardiac workup pending.

Plan
Obtain ECG and basic labs. Monitor symptoms.
"""


def generate_radiology_report() -> str:
    """Generate a synthetic radiology report."""
    modality = _fake.random_element(["Chest X-ray", "CT Abdomen", "MRI Brain"])
    finding = _fake.random_element([
        "no acute cardiopulmonary process",
        "mild bibasilar atelectasis",
        "no evidence of acute intracranial abnormality",
    ])

    return f"""Examination
{modality}

Clinical Indication
Evaluate for acute process.

Findings
The lungs are clear. No pleural effusion. Cardiac silhouette within
normal limits. {finding.capitalize()}.

Impression
{finding.capitalize()}.
"""


def generate_progress_note() -> str:
    """Generate a synthetic SOAP-format progress note."""
    return f"""Subjective
Patient reports feeling improved since yesterday. Denies new complaints.

Objective
Vital signs stable. Afebrile. Labs within normal limits.

Assessment
Improving clinically.

Plan
Continue current management. Reassess tomorrow.
"""


def synthetic_clinical_records(n: int = 4) -> list[dict[str, str]]:
    """Generate a small batch of varied synthetic clinical text records.

    Returns:
        List of dicts with keys: record_id, note_type, raw_text
    """
    generators = [
        ("Discharge Summary", generate_discharge_summary),
        ("History and Physical", generate_history_and_physical),
        ("Radiology Report", generate_radiology_report),
        ("Progress Note", generate_progress_note),
    ]
    records = []
    for i in range(n):
        note_type, gen_fn = generators[i % len(generators)]
        records.append({
            "record_id": str(uuid.uuid4()),
            "note_type": note_type,
            "raw_text": gen_fn(),
        })
    return records
