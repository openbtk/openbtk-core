"""``LocalVocabBackend``: user-supplied local vocabulary files, fully
offline, no API key (docs/03_ARCHITECTURE.md section 8.2).

**A deliberately simple, documented interchange format** -- three columns,
``code,system,display`` (a CSV header row required) -- not any official
vocabulary distribution format. Real releases (UMLS's RRF multi-file
schema, LOINC's own multi-table CSV export, SNOMED CT's RF2 format) are
complex, versioned, and vocabulary-specific; parsing each natively would
be a real, ongoing maintenance burden this project does not take on
(docs/09_CODING_STANDARDS.md section 7 -- "wrap, don't reinvent" cuts both
ways: reinventing a vocabulary-release parser here would be exactly the
"net-new code without a clear justification" that rule warns against).
This backend defines one universal format instead; a user with a real,
licensed vocabulary release converts it into this format themselves (a
few lines of their own ETL) before pointing this backend at it. ``system``
must be one of ``openbtk.core.schemas.CodeSystem``'s exact values.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

from openbtk.core.base import BaseTerminologyService
from openbtk.core.errors import TerminologyError
from openbtk.core.registry import TERMINOLOGY_REGISTRY
from openbtk.core.schemas import CodeSystem, Concept

if TYPE_CHECKING:
    from collections.abc import Mapping


@TERMINOLOGY_REGISTRY.register("terminology.general.local")
class LocalVocabBackend(BaseTerminologyService):
    """Resolve/validate codes from a user-supplied ``code,system,display``
    CSV file.

    Args:
        path: Path to the CSV file. Read lazily on first real use
            (docs/09_CODING_STANDARDS.md rule 11 -- constructors do no I/O),
            not at construction time.

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     path = Path(d) / "vocab.csv"
        ...     _ = path.write_text(
        ...         "code,system,display\\n"
        ...         "73211009,SNOMED,Diabetes mellitus\\n"
        ...     )
        ...     backend = LocalVocabBackend(path=str(path))
        ...     concept = backend.resolve("73211009", CodeSystem.SNOMED)
        >>> concept.display
        'Diabetes mellitus'
    """

    def __init__(self, *, path: str) -> None:
        self._path = path
        self._by_system: dict[CodeSystem, dict[str, str]] | None = None

    def _table(self) -> Mapping[CodeSystem, Mapping[str, str]]:
        if self._by_system is None:
            self._by_system = self._load()
        return self._by_system

    def _load(self) -> dict[CodeSystem, dict[str, str]]:
        path = Path(self._path)
        try:
            handle = path.open(newline="", encoding="utf-8")
        except OSError as e:
            raise TerminologyError(
                f"Could not open local vocabulary file {self._path}.",
                context={"path": self._path},
            ) from e
        result: dict[CodeSystem, dict[str, str]] = {}
        try:
            reader = csv.DictReader(handle)
            missing = {"code", "system", "display"} - set(reader.fieldnames or [])
            if missing:
                raise TerminologyError(
                    f"{self._path} is missing required column(s): {sorted(missing)}.",
                    context={"path": self._path},
                )
            for lineno, row in enumerate(reader, start=2):
                try:
                    system = CodeSystem(row["system"])
                except ValueError as e:
                    raise TerminologyError(
                        f"{self._path} line {lineno}: unknown system "
                        f"{row['system']!r}.",
                        context={"path": self._path, "line": lineno},
                    ) from e
                result.setdefault(system, {})[row["code"]] = row["display"]
        finally:
            handle.close()
        return result

    def resolve(self, code: str, system: CodeSystem) -> Concept | None:
        display = self._table().get(system, {}).get(code)
        if display is None:
            return None
        return Concept(code=code, system=system, display=display)

    def validate(self, code: str, system: CodeSystem) -> bool:
        return self.resolve(code, system) is not None

    def is_authoritative(self, system: CodeSystem) -> bool:
        """True only for a system the supplied file has rows for: the file is
        taken as that system's full vocabulary. A system it has no rows for is
        simply not covered, so absence there proves nothing."""
        return system in self._table()

    def map(
        self,
        code: str,  # noqa: ARG002 -- BaseTerminologyService interface, unused by design
        from_system: CodeSystem,  # noqa: ARG002
        to_system: CodeSystem,  # noqa: ARG002
    ) -> list[Concept]:
        """Always returns ``[]`` -- a flat code/display table carries no
        cross-system equivalence data. A real, disclosed limitation."""
        return []
