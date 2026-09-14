"""Regression test for T3 (docs/06_SECURITY_COMPLIANCE.md section 1, "PHI
sent to a third-party API"), closed by ``Registry.create``'s
``sends_data_offsite`` enforcement (openbtk/core/registry.py, FR-V-07).

This file used to document a real, previously undetected gap: as of M1,
``sends_data_offsite`` (base.py), ``PolicyConfig.allow_offsite_providers``
(config.py), and ``PolicyError`` (errors.py) existed as three declared,
disconnected pieces with nothing wiring them together -- the test was
written against the real construction path and marked
``xfail(strict=True)`` specifically so it would hard-fail (forcing an
update here) the moment enforcement was added, rather than being left as a
stale xfail. Enforcement has now been added directly to
``Registry.create``/``create_from_config``, so this is that update: the
``xfail`` is gone and these are real regression tests.

Uses a throwaway ``Registry`` instance (same pattern as
tests/unit/core/test_registry.py's ``loader_registry`` fixture), not the
real global ``EMBEDDING_REGISTRY``: registering a fake offsite provider
into a real global registry means the embedding contract suite sweeps it
up too (no opt-out, by design), and the contract suite's generic
``create(key)`` calls carry no policy -- which is now exactly the call
enforcement is supposed to block. A private registry instance tests the
enforcement mechanism in isolation instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pytest

from openbtk.core.base import BaseEmbeddingProvider
from openbtk.core.config import PolicyConfig
from openbtk.core.errors import PolicyError
from openbtk.core.registry import Registry

if TYPE_CHECKING:
    from numpy.typing import NDArray

_OFFSITE_KEY = "embedding.general.offsite_probe"
_LOCAL_ONLY_KEY = "embedding.general.local_probe"


class _FakeOffsiteEmbeddingProvider(BaseEmbeddingProvider):
    """Stands in for OpenAIEmbedding (docs/06_SECURITY_COMPLIANCE.md
    section 3.3's own example of a provider with sends_data_offsite=True).
    embed() is never actually exercised by these tests -- construction
    itself is the operation being policy-gated."""

    sends_data_offsite: ClassVar[bool] = True

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        return np.zeros((len(texts), self.dimension), dtype=np.float32)

    @property
    def dimension(self) -> int:
        return 1


class _FakeLocalEmbeddingProvider(BaseEmbeddingProvider):
    """A provider that does NOT send data offsite -- the negative control
    proving enforcement is targeted, not a blanket block on construction."""

    sends_data_offsite: ClassVar[bool] = False

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        return np.zeros((len(texts), self.dimension), dtype=np.float32)

    @property
    def dimension(self) -> int:
        return 1


@pytest.fixture
def embedding_registry() -> Registry[BaseEmbeddingProvider]:
    registry: Registry[BaseEmbeddingProvider] = Registry(
        "embedding",
        BaseEmbeddingProvider,  # type: ignore[type-abstract]
    )
    registry.register(_OFFSITE_KEY)(_FakeOffsiteEmbeddingProvider)
    registry.register(_LOCAL_ONLY_KEY)(_FakeLocalEmbeddingProvider)
    return registry


class TestOffsiteProviderBlockedByDefault:
    def test_no_policy_argument_blocks_construction(
        self, embedding_registry: Registry[BaseEmbeddingProvider]
    ) -> None:
        """The exact scenario base.py's docstring promises: a caller who
        passes no policy at all must still be protected -- absence of a
        policy is not treated as absence of a rule."""
        with pytest.raises(PolicyError, match="sends_data_offsite"):
            embedding_registry.create(_OFFSITE_KEY)

    def test_explicit_safe_default_policy_blocks_construction(
        self, embedding_registry: Registry[BaseEmbeddingProvider]
    ) -> None:
        policy = PolicyConfig()  # allow_offsite_providers=False, the default
        assert policy.allow_offsite_providers is False
        with pytest.raises(PolicyError):
            embedding_registry.create(_OFFSITE_KEY, policy=policy)

    def test_error_context_names_the_offending_class_and_key(
        self, embedding_registry: Registry[BaseEmbeddingProvider]
    ) -> None:
        with pytest.raises(PolicyError) as exc_info:
            embedding_registry.create(_OFFSITE_KEY)
        assert exc_info.value.context["key"] == _OFFSITE_KEY
        assert exc_info.value.context["class_name"] == "_FakeOffsiteEmbeddingProvider"

    def test_create_from_config_is_equally_enforced(
        self, embedding_registry: Registry[BaseEmbeddingProvider]
    ) -> None:
        """create_from_config is the path a real pipeline config actually
        uses -- confirming enforcement isn't bypassable just by going
        through the config-driven entry point instead of create() directly."""
        with pytest.raises(PolicyError):
            embedding_registry.create_from_config({"type": _OFFSITE_KEY})


class TestOffsiteProviderPermittedWhenExplicitlyAllowed:
    def test_explicit_opt_in_permits_construction(
        self, embedding_registry: Registry[BaseEmbeddingProvider]
    ) -> None:
        policy = PolicyConfig(allow_offsite_providers=True)
        provider = embedding_registry.create(_OFFSITE_KEY, policy=policy)
        assert isinstance(provider, _FakeOffsiteEmbeddingProvider)


class TestLocalOnlyProviderIsUnaffected:
    """Negative control: enforcement must be specific to
    sends_data_offsite=True, not an accidental blanket restriction on
    Registry.create in general."""

    def test_constructs_with_no_policy_at_all(
        self, embedding_registry: Registry[BaseEmbeddingProvider]
    ) -> None:
        provider = embedding_registry.create(_LOCAL_ONLY_KEY)
        assert isinstance(provider, _FakeLocalEmbeddingProvider)

    def test_constructs_under_the_strict_default_policy(
        self, embedding_registry: Registry[BaseEmbeddingProvider]
    ) -> None:
        provider = embedding_registry.create(_LOCAL_ONLY_KEY, policy=PolicyConfig())
        assert isinstance(provider, _FakeLocalEmbeddingProvider)
