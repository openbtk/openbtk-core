"""``CachedTerminologyService``: a TTL, content-addressed on-disk cache
wrapping any other ``BaseTerminologyService`` -- "offline operation once
warmed" (docs/03_ARCHITECTURE.md section 8.2).

**Not registered in ``TERMINOLOGY_REGISTRY``.** Its entire configuration
IS another already-registered (or directly constructed) terminology
service instance -- the same "wrap a real component by direct
composition, not by a registry string" treatment already given to
``ConceptOverlapReranker.extract_concepts`` (task 5.7) and
``RAGPipeline``'s constructor-injected providers (task 5.8), for the same
reason: nothing here is YAML-config-shaped, since a config file cannot
express "the object my ``backend`` parameter should be." A caller builds
the wrapped backend first (from the registry or directly) and passes the
real instance in.

Caches ``resolve()``/``map()`` results (``validate()`` is derived from
``resolve()`` and shares its cache, so it never makes a second real call).
A cache entry older than ``ttl_seconds`` is treated as a miss and the
wrapped backend is called again. Cache keys are content-addressed
(SHA-256 of the method name and its arguments), so cache files never leak
any of the arguments themselves into a filename.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from openbtk.core.base import BaseTerminologyService
from openbtk.core.schemas import CodeSystem, Concept


class CachedTerminologyService(BaseTerminologyService):
    """Wrap ``backend`` with a TTL disk cache under ``cache_dir``.

    Args:
        backend: The real terminology service to cache results from.
        cache_dir: Directory for cache files. Created lazily on first
            real write, not in ``__init__`` (docs/09_CODING_STANDARDS.md
            rule 11).
        ttl_seconds: How long a cached entry stays valid. Default one day.

    Example:
        >>> import tempfile
        >>> from openbtk.terminology.bundled import BundledMinimalBackend
        >>> with tempfile.TemporaryDirectory() as d:
        ...     cached = CachedTerminologyService(
        ...         backend=BundledMinimalBackend(), cache_dir=d,
        ...     )
        ...     first = cached.resolve("E11.9", CodeSystem.ICD10CM)
        ...     second = cached.resolve("E11.9", CodeSystem.ICD10CM)
        >>> first == second
        True
    """

    def __init__(
        self,
        *,
        backend: BaseTerminologyService,
        cache_dir: str,
        ttl_seconds: float = 86400.0,
    ) -> None:
        self._backend = backend
        self._cache_dir = Path(cache_dir)
        self._ttl_seconds = ttl_seconds

    def _cache_path(self, method: str, *parts: str) -> Path:
        key = "\x1f".join((method, *parts))
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self._cache_dir / f"{digest}.json"

    def _read_cache(self, path: Path) -> Any | None:
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, KeyError):
            return None
        if time.time() - payload["cached_at"] > self._ttl_seconds:
            return None
        return payload["value"]

    def _write_cache(self, path: Path, value: Any) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"cached_at": time.time(), "value": value}), encoding="utf-8"
        )

    def resolve(self, code: str, system: CodeSystem) -> Concept | None:
        path = self._cache_path("resolve", system.value, code)
        cached = self._read_cache(path)
        if cached is not None:
            concept = cached["concept"]
            return Concept.model_validate(concept) if concept else None
        result = self._backend.resolve(code, system)
        self._write_cache(
            path, {"concept": result.model_dump(mode="json") if result else None}
        )
        return result

    def validate(self, code: str, system: CodeSystem) -> bool:
        return self.resolve(code, system) is not None

    def is_authoritative(self, system: CodeSystem) -> bool:
        return self._backend.is_authoritative(system)

    def map(
        self, code: str, from_system: CodeSystem, to_system: CodeSystem
    ) -> list[Concept]:
        path = self._cache_path("map", from_system.value, to_system.value, code)
        cached = self._read_cache(path)
        if cached is not None:
            return [Concept.model_validate(c) for c in cached["concepts"]]
        result = self._backend.map(code, from_system, to_system)
        self._write_cache(
            path, {"concepts": [c.model_dump(mode="json") for c in result]}
        )
        return result
