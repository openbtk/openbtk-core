"""Unit tests for openbtk.terminology.cache.CachedTerminologyService."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from openbtk.core.base import BaseTerminologyService
from openbtk.core.schemas import CodeSystem, Concept
from openbtk.terminology.cache import CachedTerminologyService

if TYPE_CHECKING:
    from pathlib import Path


class _CountingBackend(BaseTerminologyService):
    """A real, minimal BaseTerminologyService that counts calls -- proves
    the cache actually avoids calling through, not merely that it returns
    a plausible-looking value."""

    def __init__(self) -> None:
        self.resolve_calls = 0
        self.map_calls = 0

    def resolve(self, code: str, system: CodeSystem) -> Concept | None:
        self.resolve_calls += 1
        if code == "known":
            return Concept(code=code, system=system, display="Known Concept")
        return None

    def validate(self, code: str, system: CodeSystem) -> bool:
        return self.resolve(code, system) is not None

    def map(
        self, code: str, from_system: CodeSystem, to_system: CodeSystem
    ) -> list[Concept]:
        self.map_calls += 1
        return [Concept(code=code, system=to_system, display="Mapped")]


class TestResolveCaching:
    def test_second_call_does_not_hit_the_backend(self, tmp_path: Path) -> None:
        backend = _CountingBackend()
        cached = CachedTerminologyService(backend=backend, cache_dir=str(tmp_path))
        first = cached.resolve("known", CodeSystem.SNOMED)
        second = cached.resolve("known", CodeSystem.SNOMED)
        assert first == second
        assert backend.resolve_calls == 1

    def test_caches_a_negative_result_too(self, tmp_path: Path) -> None:
        backend = _CountingBackend()
        cached = CachedTerminologyService(backend=backend, cache_dir=str(tmp_path))
        assert cached.resolve("unknown", CodeSystem.SNOMED) is None
        assert cached.resolve("unknown", CodeSystem.SNOMED) is None
        assert backend.resolve_calls == 1

    def test_different_codes_are_cached_separately(self, tmp_path: Path) -> None:
        backend = _CountingBackend()
        cached = CachedTerminologyService(backend=backend, cache_dir=str(tmp_path))
        cached.resolve("known", CodeSystem.SNOMED)
        cached.resolve("known", CodeSystem.ICD10CM)
        assert backend.resolve_calls == 2

    def test_expired_entry_is_refetched(self, tmp_path: Path) -> None:
        backend = _CountingBackend()
        cached = CachedTerminologyService(
            backend=backend, cache_dir=str(tmp_path), ttl_seconds=0.01
        )
        cached.resolve("known", CodeSystem.SNOMED)
        time.sleep(0.02)
        cached.resolve("known", CodeSystem.SNOMED)
        assert backend.resolve_calls == 2

    def test_corrupt_cache_file_is_treated_as_a_miss(self, tmp_path: Path) -> None:
        backend = _CountingBackend()
        cached = CachedTerminologyService(backend=backend, cache_dir=str(tmp_path))
        cached.resolve("known", CodeSystem.SNOMED)
        for f in tmp_path.glob("*.json"):
            f.write_text("not json", encoding="utf-8")
        cached.resolve("known", CodeSystem.SNOMED)
        assert backend.resolve_calls == 2

    def test_validate_shares_the_resolve_cache(self, tmp_path: Path) -> None:
        backend = _CountingBackend()
        cached = CachedTerminologyService(backend=backend, cache_dir=str(tmp_path))
        cached.resolve("known", CodeSystem.SNOMED)
        assert cached.validate("known", CodeSystem.SNOMED) is True
        assert backend.resolve_calls == 1


class TestAuthorityIsDelegated:
    def test_a_complete_backend_stays_authoritative_through_the_cache(
        self, tmp_path: Path
    ) -> None:
        cached = CachedTerminologyService(
            backend=_CountingBackend(), cache_dir=str(tmp_path)
        )
        assert cached.is_authoritative(CodeSystem.SNOMED) is True  # the default

    def test_a_partial_backend_stays_partial_through_the_cache(
        self, tmp_path: Path
    ) -> None:
        from openbtk.terminology.bundled import BundledMinimalBackend

        cached = CachedTerminologyService(
            backend=BundledMinimalBackend(), cache_dir=str(tmp_path)
        )
        assert cached.is_authoritative(CodeSystem.ICD10CM) is False


class TestMapCaching:
    def test_second_call_does_not_hit_the_backend(self, tmp_path: Path) -> None:
        backend = _CountingBackend()
        cached = CachedTerminologyService(backend=backend, cache_dir=str(tmp_path))
        first = cached.map("X", CodeSystem.SNOMED, CodeSystem.RXNORM)
        second = cached.map("X", CodeSystem.SNOMED, CodeSystem.RXNORM)
        assert first == second
        assert backend.map_calls == 1


class TestCacheFiles:
    def test_cache_directory_is_created_lazily(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "does-not-exist-yet"
        backend = _CountingBackend()
        cached = CachedTerminologyService(backend=backend, cache_dir=str(cache_dir))
        assert not cache_dir.exists()
        cached.resolve("known", CodeSystem.SNOMED)
        assert cache_dir.is_dir()

    def test_cache_file_is_content_addressed_json(self, tmp_path: Path) -> None:
        backend = _CountingBackend()
        cached = CachedTerminologyService(backend=backend, cache_dir=str(tmp_path))
        cached.resolve("known", CodeSystem.SNOMED)
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        assert "cached_at" in payload
        assert payload["value"]["concept"]["code"] == "known"
