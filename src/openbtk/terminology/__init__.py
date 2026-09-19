"""Terminology resolution across SNOMED CT, LOINC, RxNorm, ICD-10-CM and
UMLS (docs/03_ARCHITECTURE.md section 8.2).

Licence-aware: never bundles restricted vocabularies. Three backends:
``UMLSRestBackend`` (real UMLS UTS REST API, requires a user's own
licence + API key), ``LocalVocabBackend`` (user-supplied CSV, fully
offline), ``BundledMinimalBackend`` (a small, verified, permissively-
licensed ICD-10-CM subset shipped with the package). ``CachedTerminologyService``
(``openbtk.terminology.cache``) wraps any of the three with a TTL,
content-addressed disk cache for offline operation once warmed.

Submodules are imported here (registering the three backends as a side
effect) for the same reason every other component package does: none of
them need a heavy optional dependency merely to be *defined* (``httpx`` is
already a core dependency; the local/bundled backends use only the
standard library).
"""

from __future__ import annotations

from openbtk.terminology import bundled, local, umls

__all__ = ["bundled", "local", "umls"]
