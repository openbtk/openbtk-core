"""
Clinical text dataset adapters.

- MIMICNotesDataset: credentialed MIMIC-III/IV note loading
- PubMedDataset: open PubMed abstract search via NCBI E-utilities
- SyntheaNotesDataset: synthetic notes for testing (no credentials required)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from opentbtk.core.base import BaseDatasetAdapter
from opentbtk.core.errors import DatasetError
from opentbtk.core.registry import DATASET_REGISTRY
from .loaders import MIMICNotesLoader
from .schemas import ClinicalTextRecord

log = structlog.get_logger(__name__)


@DATASET_REGISTRY.register("dataset.clinical_text.mimic_notes")
class MIMICNotesDataset(BaseDatasetAdapter):
    """Load clinical notes from a credentialed local MIMIC-III/IV export.

    MIMIC-III/IV require PhysioNet credentialed access and a completed
    data use agreement (CITI training required). This adapter does NOT
    download or bundle any MIMIC data — it loads from a user-supplied local
    path obtained through authorized PhysioNet access.

    See: https://physionet.org/content/mimiciv/ for access requirements.

    Args:
        version: "iii" or "iv" — determines expected CSV column names.
    """

    def __init__(self, version: str = "iv") -> None:
        if version not in ("iii", "iv"):
            raise DatasetError(f"Unsupported MIMIC version '{version}'. Use 'iii' or 'iv'.")
        self._version = version

    @property
    def name(self) -> str:
        return f"MIMIC-{self._version.upper()} Notes"

    @property
    def requires_credentials(self) -> bool:
        return True

    @property
    def license(self) -> str:
        return "PhysioNet Credentialed Health Data License — https://physionet.org/content/mimiciv/"

    def load(self, path: str | None = None, **kwargs: Any) -> list[ClinicalTextRecord]:
        """Load MIMIC notes from a local credentialed export.

        Args:
            path: Local path to the notes CSV (e.g., NOTEEVENTS.csv for
                MIMIC-III, or discharge.csv for MIMIC-IV-Note).
            **kwargs: Passed through to MIMICNotesLoader (id_column,
                text_column, category_column, hash_salt).

        Returns:
            List of ClinicalTextRecord objects with hashed/pseudonymous IDs.

        Raises:
            DatasetError: If `path` is not provided or the file is missing,
                with guidance on obtaining credentialed access.
        """
        if path is None:
            raise DatasetError(
                "MIMICNotesDataset requires a local 'path' to credentialed "
                "MIMIC data. This dataset cannot be auto-downloaded — see "
                "https://physionet.org/content/mimiciv/ to request access.",
            )
        if not Path(path).exists():
            raise DatasetError(
                f"MIMIC notes file not found at: {path}. Ensure you have "
                "completed PhysioNet credentialing and provided the correct "
                "local path to your downloaded export.",
                context={"path": path},
            )

        loader = MIMICNotesLoader(**kwargs)
        records = loader.load_all(path)
        log.info(
            "dataset.loaded",
            dataset="mimic_notes",
            version=self._version,
            n_records=len(records),
        )
        return records


@DATASET_REGISTRY.register("dataset.clinical_text.pubmed")
class PubMedDataset(BaseDatasetAdapter):
    """Search and load PubMed abstracts via NCBI E-utilities.

    Wraps Biopython's `Bio.Entrez` module. Open access — no credentials
    required, but NCBI requests an email address for API courtesy/rate
    limiting and recommends an API key for higher request rates.

    Args:
        email: Contact email required by NCBI E-utilities usage policy.
        api_key: Optional NCBI API key for higher rate limits
            (10 req/s vs 3 req/s without).
    """

    def __init__(self, email: str, api_key: str | None = None) -> None:
        if not email:
            raise DatasetError(
                "PubMedDataset requires an 'email' argument per NCBI "
                "E-utilities usage policy (https://www.ncbi.nlm.nih.gov/books/NBK25497/).",
            )
        self._email = email
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "PubMed Abstracts"

    @property
    def requires_credentials(self) -> bool:
        return False

    @property
    def license(self) -> str:
        return "NCBI Public Domain / Publisher copyright varies — see PubMed terms"

    def load(
        self,
        query: str = "",
        max_results: int = 20,
        date_range: tuple[str, str] | None = None,
        **kwargs: Any,
    ) -> list[ClinicalTextRecord]:
        """Search PubMed and load matching abstracts.

        Args:
            query: PubMed search query (supports standard PubMed query syntax,
                e.g., 'diabetes AND machine learning').
            max_results: Maximum number of records to retrieve.
            date_range: Optional (start_date, end_date) in "YYYY/MM/DD" format.

        Returns:
            List of ClinicalTextRecord objects, one per abstract.

        Raises:
            DatasetError: On E-utilities API failure or missing biopython.
        """
        try:
            from Bio import Entrez
        except ImportError as e:
            raise DatasetError(
                "biopython is required for PubMedDataset. "
                "Install with: pip install opentbtk[clinical_text]",
            ) from e

        Entrez.email = self._email
        if self._api_key:
            Entrez.api_key = self._api_key

        search_term = query
        if date_range:
            search_term += f' AND ("{date_range[0]}"[Date - Publication] : "{date_range[1]}"[Date - Publication])'

        try:
            search_handle = Entrez.esearch(
                db="pubmed", term=search_term, retmax=max_results
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()
            id_list = search_results.get("IdList", [])

            if not id_list:
                log.info("dataset.pubmed.no_results", query=query)
                return []

            fetch_handle = Entrez.efetch(
                db="pubmed", id=id_list, rettype="abstract", retmode="xml"
            )
            records_xml = Entrez.read(fetch_handle)
            fetch_handle.close()
        except Exception as e:
            raise DatasetError(
                "PubMed E-utilities request failed",
                context={"query": query},
            ) from e

        records: list[ClinicalTextRecord] = []
        for article in records_xml.get("PubmedArticle", []):
            try:
                medline = article["MedlineCitation"]
                pmid = str(medline["PMID"])
                article_data = medline["Article"]
                title = str(article_data.get("ArticleTitle", ""))
                abstract_parts = article_data.get("Abstract", {}).get("AbstractText", [])
                abstract_text = " ".join(str(p) for p in abstract_parts)
                full_text = f"{title}\n\n{abstract_text}".strip()
                if not full_text:
                    continue
                records.append(ClinicalTextRecord(
                    record_id=f"pmid_{pmid}",
                    source="pubmed",
                    note_type="Abstract",
                    raw_text=full_text,
                    metadata={"pmid": pmid, "title": title},
                ))
            except (KeyError, TypeError) as e:
                log.warning("dataset.pubmed.parse_skip", error=str(e))
                continue

        log.info("dataset.loaded", dataset="pubmed", n_records=len(records))
        return records


@DATASET_REGISTRY.register("dataset.clinical_text.synthea_notes")
class SyntheaNotesDataset(BaseDatasetAdapter):
    """Generate synthetic clinical notes for testing — no credentials required.

    Uses OpenTBTK's built-in synthetic note templates (the same generators
    used in test fixtures) rather than the full Java-based Synthea engine,
    making this lightweight and dependency-free. Suitable for unit tests,
    demos, and CI — NOT for research validity claims.

    For full synthetic *population* generation with realistic epidemiology,
    use the actual Synthea project (https://github.com/synthetichealth/synthea)
    and load its output FHIR Bundles via `opentbtk.data.ehr_emr` adapters.
    """

    @property
    def name(self) -> str:
        return "Synthea-style Synthetic Notes (lightweight)"

    @property
    def requires_credentials(self) -> bool:
        return False

    @property
    def license(self) -> str:
        return "Synthetic data — no license restrictions"

    def load(self, n: int = 10, seed: int = 42, **kwargs: Any) -> list[ClinicalTextRecord]:
        """Generate n synthetic clinical text records.

        Args:
            n: Number of synthetic records to generate.
            seed: Random seed for reproducibility.

        Returns:
            List of synthetic ClinicalTextRecord objects.
        """
        import uuid
        from faker import Faker

        fake = Faker()
        Faker.seed(seed)

        note_templates = [
            self._discharge_summary_template,
            self._hp_note_template,
            self._radiology_report_template,
        ]

        records: list[ClinicalTextRecord] = []
        for i in range(n):
            template_fn = note_templates[i % len(note_templates)]
            note_type, text = template_fn(fake)
            records.append(ClinicalTextRecord(
                record_id=str(uuid.uuid4()),
                source="synthea_notes_synthetic",
                note_type=note_type,
                raw_text=text,
                metadata={"synthetic": True},
            ))

        log.info("dataset.loaded", dataset="synthea_notes", n_records=len(records))
        return records

    @staticmethod
    def _discharge_summary_template(fake: Any) -> tuple[str, str]:
        condition = fake.random_element([
            "pneumonia", "heart failure exacerbation", "diabetic ketoacidosis",
        ])
        return "Discharge Summary", (
            f"History of Present Illness\nPatient admitted with {condition}.\n\n"
            f"Assessment and Plan\nImproved with treatment. Discharged in stable condition."
        )

    @staticmethod
    def _hp_note_template(fake: Any) -> tuple[str, str]:
        complaint = fake.random_element(["chest pain", "abdominal pain", "headache"])
        return "History and Physical", (
            f"Chief Complaint\n{complaint.capitalize()}.\n\n"
            f"Assessment\nFurther workup pending."
        )

    @staticmethod
    def _radiology_report_template(fake: Any) -> tuple[str, str]:
        modality = fake.random_element(["Chest X-ray", "CT Abdomen"])
        return "Radiology Report", (
            f"Examination\n{modality}\n\n"
            f"Impression\nNo acute abnormality identified."
        )
