"""Credentialed clinical-text dataset adapters (docs/05_DATA_MODALITY_SPEC.md
section 1.5).

``N2C2DeidDataset`` reads the i2b2/n2c2 2014 de-identification challenge
corpus from a directory the user already has -- and **never downloads
anything**. That data is released only under a Data Use Agreement; an adapter
that fetched it, or fell back to a substitute when it was absent, would be
exactly the "auto-download of restricted data" section 1.5 forbids. With no
path (or a path that is not a directory) ``load`` raises ``DatasetError``
naming the registration page.

**Format, disclosed with its verification status.** Each ``*.xml`` file is an
``<deIdi2b2>`` document: a ``<TEXT>`` element (CDATA) and a ``<TAGS>`` element
whose children -- ``NAME``, ``LOCATION``, ``DATE``, ``CONTACT``, ``ID``,
``AGE``, ``PROFESSION`` -- each carry ``start``, ``end``, ``text`` and
``TYPE`` attributes. This parser was written from the published annotation
scheme and has **not** been run against the real (DUA-restricted) corpus,
which this project's development environment does not have; it is exercised
against hand-built files in that shape. Two things protect against a silent
misreading of real data:

* every tag's ``text`` attribute is checked against ``TEXT[start:end]``; a
  mismatch (wrong offset convention, or line-ending normalisation by the XML
  parser shifting offsets) raises ``DatasetError`` naming the file rather than
  scoring misaligned spans;
* an element/``TYPE`` pair outside the mapping below is *counted* in
  ``LabelledDocument.unscored_spans``, never dropped silently.

**Mapping to Safe Harbor** (``PHICategory``): ``PROFESSION`` and ``AGE`` (Safe
Harbor covers only ages over 89), and the ``HOSPITAL``/``ORGANIZATION``
location types (an organisation is not a Safe Harbor identifier), have no
category and are unscored. Everything else maps as in ``_TAG_TO_CATEGORY``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openbtk.core.base import BaseDatasetAdapter
from openbtk.core.errors import DatasetError
from openbtk.core.logging import get_logger
from openbtk.core.registry import DATASET_REGISTRY
from openbtk.deid.labelled import LabelledDocument, LabelledSpan
from openbtk.deid.schemas import PHICategory

if TYPE_CHECKING:
    from collections.abc import Iterator

log = get_logger(__name__)

_REGISTRATION_URL = "https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/"

# (element name, TYPE attribute) -> Safe Harbor category. A pair absent from
# this table is counted as unscored, never guessed at.
_TAG_TO_CATEGORY: dict[tuple[str, str], PHICategory] = {
    ("NAME", "PATIENT"): PHICategory.NAME,
    ("NAME", "DOCTOR"): PHICategory.NAME,
    ("NAME", "USERNAME"): PHICategory.NAME,
    ("LOCATION", "STREET"): PHICategory.GEOGRAPHIC_SUBDIVISION,
    ("LOCATION", "CITY"): PHICategory.GEOGRAPHIC_SUBDIVISION,
    ("LOCATION", "STATE"): PHICategory.GEOGRAPHIC_SUBDIVISION,
    ("LOCATION", "COUNTRY"): PHICategory.GEOGRAPHIC_SUBDIVISION,
    ("LOCATION", "ZIP"): PHICategory.GEOGRAPHIC_SUBDIVISION,
    ("LOCATION", "OTHER"): PHICategory.GEOGRAPHIC_SUBDIVISION,
    ("DATE", "DATE"): PHICategory.DATE,
    ("CONTACT", "PHONE"): PHICategory.PHONE_NUMBER,
    ("CONTACT", "FAX"): PHICategory.FAX_NUMBER,
    ("CONTACT", "EMAIL"): PHICategory.EMAIL,
    ("CONTACT", "URL"): PHICategory.URL,
    ("CONTACT", "IPADDRESS"): PHICategory.IP_ADDRESS,
    ("ID", "MEDICALRECORD"): PHICategory.MEDICAL_RECORD_NUMBER,
    ("ID", "HEALTHPLAN"): PHICategory.HEALTH_PLAN_BENEFICIARY_NUMBER,
    ("ID", "ACCOUNT"): PHICategory.ACCOUNT_NUMBER,
    ("ID", "LICENSE"): PHICategory.CERTIFICATE_LICENSE_NUMBER,
    ("ID", "VEHICLE"): PHICategory.VEHICLE_IDENTIFIER,
    ("ID", "DEVICE"): PHICategory.DEVICE_IDENTIFIER,
    ("ID", "BIOID"): PHICategory.BIOMETRIC_IDENTIFIER,
    ("ID", "IDNUM"): PHICategory.OTHER_UNIQUE_IDENTIFIER,
}


@DATASET_REGISTRY.register("dataset.clinical_text.n2c2_deid")
class N2C2DeidDataset(BaseDatasetAdapter):
    """i2b2/n2c2 2014 de-identification corpus, from a user-supplied directory.

    Args:
        path: Directory of ``*.xml`` files. May instead be passed to
            :meth:`load`. Never fetched or created by this class.

    Example:
        >>> import contextlib, io, tempfile
        >>> from pathlib import Path
        >>> xml = (
        ...     '<?xml version="1.0"?><deIdi2b2><TEXT><![CDATA[Seen 2090-01-05.]]>'
        ...     '</TEXT><TAGS><DATE id="P0" start="5" end="15" text="2090-01-05" '
        ...     'TYPE="DATE" comment=""/></TAGS></deIdi2b2>'
        ... )
        >>> with tempfile.TemporaryDirectory() as d:
        ...     _ = (Path(d) / "doc1.xml").write_text(xml)
        ...     with contextlib.redirect_stdout(io.StringIO()):
        ...         docs = list(N2C2DeidDataset(path=d).load())
        >>> docs[0].spans[0].category.value
        'date'
    """

    def __init__(self, *, path: str | None = None) -> None:
        self._path = path

    @property
    def name(self) -> str:
        return "n2c2 2014 De-identification (i2b2)"

    @property
    def license(self) -> str:
        return _REGISTRATION_URL

    @property
    def requires_credentials(self) -> bool:
        return True

    def load(self, **kwargs: Any) -> Iterator[LabelledDocument]:
        """Validate the source now, then stream documents lazily.

        Raises:
            DatasetError: If no directory was supplied or it does not exist
                (the message names the registration page -- this class never
                downloads), or, while iterating, if a file is malformed or
                its tag offsets do not match its text.
        """
        path = kwargs.get("path", self._path)
        directory = Path(path) if path else None
        if directory is None or not directory.is_dir():
            raise DatasetError(
                "The n2c2 de-identification corpus is released only under a "
                "Data Use Agreement and is never downloaded automatically. "
                f"Register at {_REGISTRATION_URL}, obtain the data, and pass "
                "its directory as `path`.",
                context={"dataset": "n2c2_deid", "registration": _REGISTRATION_URL},
            )
        log.info("dataset.start", dataset="n2c2_deid", source=str(directory))
        return self._iter_documents(directory)

    def _iter_documents(self, directory: Path) -> Iterator[LabelledDocument]:
        for file in sorted(directory.glob("*.xml")):
            yield _parse_document(file)


def _parse_document(file: Path) -> LabelledDocument:
    ctx = {"dataset": "n2c2_deid", "filename": file.name}
    try:
        raw = file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise DatasetError(f"Could not read {file.name}.", context=ctx) from e
    # stdlib ElementTree expands internal entities; a corpus file has no
    # business declaring any, so refuse rather than parse it.
    if "<!DOCTYPE" in raw or "<!ENTITY" in raw:
        raise DatasetError(
            f"{file.name} declares a DOCTYPE/ENTITY, which this reader refuses "
            "to parse.",
            context=ctx,
        )
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        raise DatasetError(f"{file.name} is not well-formed XML.", context=ctx) from e
    text_el = root.find("TEXT")
    if text_el is None or text_el.text is None:
        raise DatasetError(f"{file.name} has no <TEXT> element.", context=ctx)
    text = text_el.text
    spans: list[LabelledSpan] = []
    unscored = 0
    tags_el = root.find("TAGS")
    for tag in list(tags_el) if tags_el is not None else []:
        try:
            start, end = int(tag.attrib["start"]), int(tag.attrib["end"])
        except (KeyError, ValueError) as e:
            raise DatasetError(
                f"{file.name}: a <{tag.tag}> tag lacks integer start/end.",
                context=ctx,
            ) from e
        claimed = tag.attrib.get("text")
        if claimed is not None and text[start:end] != claimed:
            raise DatasetError(
                f"{file.name}: a <{tag.tag}> tag's text does not match "
                f"TEXT[{start}:{end}] -- offsets are misaligned (wrong offset "
                "convention, or line endings normalised on parse). Refusing to "
                "score misaligned spans.",
                context={**ctx, "tag": tag.tag, "start": start, "end": end},
            )
        category = _TAG_TO_CATEGORY.get((tag.tag, tag.attrib.get("TYPE", "")))
        if category is None:
            unscored += 1
            continue
        spans.append(LabelledSpan(category=category, start=start, end=end))
    return LabelledDocument(
        document_id=file.stem, text=text, spans=spans, unscored_spans=unscored
    )
