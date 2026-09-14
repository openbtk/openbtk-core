"""Synthetic, labelled PHI corpus: ground-truth spans for de-identification
accuracy regression (tests/accuracy/) and PHI-leak testing (tests/security/).

docs/07_TEST_CHARTER.md section 4 names this fixture explicitly:
``labelled_phi_corpus`` -- "Templates with known ground-truth PHI spans."
The roadmap calls this task "gating everything else" in M2: without ground
truth, recognizer/engine accuracy is unmeasurable.

No real PHI, ever (CLAUDE.md rule 3). Every value here comes from ``Faker``
or a hand-built synthetic-format generator, seeded for reproducibility --
the accuracy baseline in tests/accuracy/ would be meaningless against a
corpus that changes between runs.

Covers 16 of the 18 ``PHICategory`` members. ``FULL_FACE_PHOTO`` and
``BIOMETRIC_IDENTIFIER`` are not text-representable at all (see
``openbtk.deid.schemas.PHICategory``'s own docstring) and are intentionally
absent from every document this module generates.

This module is plain Python with no pytest dependency, importable
standalone (e.g. by a future benchmark harness); the pytest fixture that
wraps it for test use lives in ``tests/conftest.py``.
"""

from __future__ import annotations

import random
import uuid
from typing import TYPE_CHECKING, Final

from faker import Faker
from pydantic import BaseModel, ConfigDict, Field

from openbtk.deid.schemas import PHICategory

if TYPE_CHECKING:
    from collections.abc import Callable

# Fixed, never randomised at collection time: tests/accuracy/'s F1 baseline
# is checked into the repo against this exact corpus. Changing this seed is
# equivalent to changing the corpus and requires re-baselining deliberately
# (docs/07_TEST_CHARTER.md section 3.6: "Updating a baseline requires a PR
# that states why").
DEFAULT_SEED: Final = 20260914
DEFAULT_N_DOCUMENTS: Final = 25


class LabelledSpan(BaseModel):
    """One planted, ground-truth PHI span within a ``LabelledDocument``.

    Exists only under tests/ -- never imported from src/. ``value``
    deliberately holds the synthetic PHI substring itself: leak tests need
    something concrete to search for in a manifest, report, or log blob.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    category: PHICategory
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
    value: str = Field(..., min_length=1)


class LabelledDocument(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(..., min_length=1)
    text: str
    spans: list[LabelledSpan]


class LabelledPHICorpus(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    documents: list[LabelledDocument] = Field(..., min_length=1)

    @property
    def all_identifiers(self) -> list[str]:
        """Every planted PHI value across every document -- what a leak
        test (docs/07_TEST_CHARTER.md section 3.5) searches a manifest,
        report, or log blob for."""
        return [span.value for doc in self.documents for span in doc.spans]

    @property
    def categories_present(self) -> set[PHICategory]:
        return {span.category for doc in self.documents for span in doc.spans}


def _digits(rng: random.Random, n: int) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(n))


def _alnum(rng: random.Random, n: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(rng.choice(alphabet) for _ in range(n))


def _phone(rng: random.Random) -> str:
    # (NXX) NXX-XXXX, N in [2-9] for the leading digit of each group -- a
    # deliberately clean, single format. Real-world diversity would need
    # its own recognizer accuracy work; this is enough to exercise the
    # ensemble at M2.
    area = rng.randint(2, 9) * 100 + rng.randint(0, 99)
    prefix = rng.randint(2, 9) * 100 + rng.randint(0, 99)
    line = _digits(rng, 4)
    return f"({area}) {prefix}-{line}"


def _date(rng: random.Random) -> str:
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    year = rng.randint(1930, 2015)
    return f"{month:02d}/{day:02d}/{year}"


_GENERATORS: dict[PHICategory, Callable[[Faker, random.Random], str]] = {
    PHICategory.NAME: lambda fake, rng: str(fake.name()),
    PHICategory.GEOGRAPHIC_SUBDIVISION: lambda fake, rng: str(fake.street_address()),
    PHICategory.DATE: lambda fake, rng: _date(rng),
    PHICategory.PHONE_NUMBER: lambda fake, rng: _phone(rng),
    PHICategory.FAX_NUMBER: lambda fake, rng: _phone(rng),
    PHICategory.EMAIL: lambda fake, rng: str(fake.email()),
    PHICategory.SSN: lambda fake, rng: str(fake.ssn()),
    PHICategory.MEDICAL_RECORD_NUMBER: lambda fake, rng: f"MRN-{_digits(rng, 7)}",
    PHICategory.HEALTH_PLAN_BENEFICIARY_NUMBER: (
        lambda fake, rng: f"{_alnum(rng, 2)}{_digits(rng, 6)}"
    ),
    PHICategory.ACCOUNT_NUMBER: lambda fake, rng: _digits(rng, 10),
    PHICategory.CERTIFICATE_LICENSE_NUMBER: lambda fake, rng: f"LIC-{_alnum(rng, 6)}",
    PHICategory.VEHICLE_IDENTIFIER: (
        lambda fake, rng: f"{_alnum(rng, 3)}-{_digits(rng, 4)}"
    ),
    PHICategory.DEVICE_IDENTIFIER: lambda fake, rng: f"SN-{_alnum(rng, 8)}",
    PHICategory.URL: lambda fake, rng: str(fake.url()),
    PHICategory.IP_ADDRESS: lambda fake, rng: str(fake.ipv4()),
    PHICategory.OTHER_UNIQUE_IDENTIFIER: (
        lambda fake, rng: f"ID-{uuid.UUID(int=rng.getrandbits(128)).hex[:8]}"
    ),
}

# One comprehensive template: every text-representable category appears in
# every document, satisfying the roadmap's "all 18 categories detected on
# the synthetic corpus" exit criterion (18 minus the 2 non-text categories
# documented in PHICategory's own docstring). Order is fixed rather than
# shuffled per document -- a documented simplification, not a limitation
# that affects correctness of accuracy measurement.
_TEMPLATE: list[str | PHICategory] = [
    "Patient: ",
    PHICategory.NAME,
    "\nDOB: ",
    PHICategory.DATE,
    "\nMRN: ",
    PHICategory.MEDICAL_RECORD_NUMBER,
    "  Account #: ",
    PHICategory.ACCOUNT_NUMBER,
    "\nAddress: ",
    PHICategory.GEOGRAPHIC_SUBDIVISION,
    "\nPhone: ",
    PHICategory.PHONE_NUMBER,
    "  Fax: ",
    PHICategory.FAX_NUMBER,
    "  Email: ",
    PHICategory.EMAIL,
    "\nSSN: ",
    PHICategory.SSN,
    "  Health Plan ID: ",
    PHICategory.HEALTH_PLAN_BENEFICIARY_NUMBER,
    "\nVehicle: ",
    PHICategory.VEHICLE_IDENTIFIER,
    "  Device SN: ",
    PHICategory.DEVICE_IDENTIFIER,
    "\nLicense: ",
    PHICategory.CERTIFICATE_LICENSE_NUMBER,
    "\nPortal: ",
    PHICategory.URL,
    "  Last login IP: ",
    PHICategory.IP_ADDRESS,
    "\nReference ID: ",
    PHICategory.OTHER_UNIQUE_IDENTIFIER,
    "\nNotes: Patient presented for routine follow-up. No acute distress noted.\n",
]


def _build_document(
    document_id: str, fake: Faker, rng: random.Random
) -> LabelledDocument:
    pieces: list[str] = []
    spans: list[LabelledSpan] = []
    offset = 0
    for segment in _TEMPLATE:
        if isinstance(segment, PHICategory):
            value = _GENERATORS[segment](fake, rng)
            spans.append(
                LabelledSpan(
                    category=segment,
                    start=offset,
                    end=offset + len(value),
                    value=value,
                )
            )
            pieces.append(value)
            offset += len(value)
        else:
            pieces.append(segment)
            offset += len(segment)
    return LabelledDocument(document_id=document_id, text="".join(pieces), spans=spans)


def build_labelled_phi_corpus(
    *, n_documents: int = DEFAULT_N_DOCUMENTS, seed: int = DEFAULT_SEED
) -> LabelledPHICorpus:
    """Build a deterministic, synthetic, labelled PHI corpus.

    Args:
        n_documents: Number of documents to generate.
        seed: Seed for both Faker and the stdlib RNG used for hand-built
            formats. The same seed always produces the same corpus.

    Example:
        >>> corpus = build_labelled_phi_corpus(n_documents=2, seed=1)
        >>> len(corpus.documents)
        2
        >>> corpus.documents[0].spans[0].value in corpus.documents[0].text
        True
    """
    fake = Faker()
    fake.seed_instance(seed)
    rng = random.Random(seed)
    documents = [_build_document(f"doc-{i}", fake, rng) for i in range(n_documents)]
    return LabelledPHICorpus(documents=documents)
