"""
Shared pytest fixtures for openbtk tests.

All fixtures here use synthetic data ONLY. Never reference real PHI,
real datasets requiring credentials, or real patient identifiers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow `from fixtures.synthetic_text import ...` in test files
sys.path.insert(0, str(Path(__file__).parent))

from fixtures.synthetic_text import (  # noqa: E402
    generate_discharge_summary,
    generate_history_and_physical,
    generate_radiology_report,
    generate_progress_note,
    synthetic_clinical_records,
)


@pytest.fixture(scope="session")
def discharge_summary_text() -> str:
    """A single synthetic discharge summary as raw text."""
    return generate_discharge_summary()


@pytest.fixture(scope="session")
def hp_note_text() -> str:
    """A single synthetic History & Physical note."""
    return generate_history_and_physical()


@pytest.fixture(scope="session")
def radiology_report_text() -> str:
    """A single synthetic radiology report."""
    return generate_radiology_report()


@pytest.fixture(scope="session")
def progress_note_text() -> str:
    """A single synthetic SOAP progress note."""
    return generate_progress_note()


@pytest.fixture
def clinical_text_record_fixture():
    """A single ClinicalTextRecord built from synthetic discharge summary."""
    from openbtk.data.clinical_text.schemas import ClinicalTextRecord
    import uuid

    return ClinicalTextRecord(
        record_id=str(uuid.uuid4()),
        source="synthetic",
        note_type="Discharge Summary",
        raw_text=generate_discharge_summary(),
    )


@pytest.fixture
def clinical_text_records_batch():
    """A batch of varied synthetic ClinicalTextRecord objects."""
    from openbtk.data.clinical_text.schemas import ClinicalTextRecord

    raw = synthetic_clinical_records(n=4)
    return [
        ClinicalTextRecord(
            record_id=r["record_id"],
            source="synthetic",
            note_type=r["note_type"],
            raw_text=r["raw_text"],
        )
        for r in raw
    ]


@pytest.fixture(autouse=True)
def _no_real_network_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Safety net: block accidental outbound HTTP calls in unit tests.

    Integration tests that legitimately need network access should override
    this fixture locally or be marked with @pytest.mark.integration and
    excluded from default test runs.
    """
    # Intentionally permissive placeholder — tighten per-project as needed.
    # Example tightening:
    # monkeypatch.setattr("httpx.Client.request", _raise_on_network_call)
    pass
