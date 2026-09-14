"""Shared contract every registered BaseEmbeddingProvider must satisfy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from openbtk.core.registry import EMBEDDING_REGISTRY

if TYPE_CHECKING:
    from openbtk.core.base import BaseEmbeddingProvider


def _new_instance(key: str) -> BaseEmbeddingProvider:
    return EMBEDDING_REGISTRY.create(key)


@pytest.mark.parametrize("key", EMBEDDING_REGISTRY.list_keys())
class TestEmbeddingContract:
    def test_declares_sends_data_offsite(self, key: str) -> None:
        """Every provider must declare this -- it is what FR-V-07's policy
        enforcement (allow_offsite_providers) reads to decide whether
        construction is even permitted under a local-only policy."""
        provider = _new_instance(key)
        assert isinstance(provider.sends_data_offsite, bool)

    def test_embed_shape_and_dtype(self, key: str) -> None:
        provider = _new_instance(key)
        texts = ["one", "two", "three"]
        vectors = provider.embed(texts)
        assert vectors.shape == (len(texts), provider.dimension)
        assert vectors.dtype == np.float32

    def test_embed_one_matches_first_row_of_embed(self, key: str) -> None:
        provider = _new_instance(key)
        text = "hello"
        one = provider.embed_one(text)
        batch = provider.embed([text])
        assert one.shape == (provider.dimension,)
        np.testing.assert_array_equal(one, batch[0])

    def test_dimension_is_positive(self, key: str) -> None:
        provider = _new_instance(key)
        assert provider.dimension > 0

    def test_batch_size_is_positive(self, key: str) -> None:
        provider = _new_instance(key)
        assert provider.batch_size > 0

    def test_model_identity_default_raises_unless_overridden(self, key: str) -> None:
        """The base default is NotImplementedError, not a fabricated
        placeholder identity (docs/03_ARCHITECTURE.md section 4.1's
        reasoning: a wrong default is worse than an explicit failure). A
        provider MAY override this with a real ModelIdentity; both outcomes
        are contract-valid, so this only asserts against a silent, wrong
        default slipping through undetected."""
        provider = _new_instance(key)
        try:
            identity = provider.model_identity()
        except NotImplementedError:
            return
        assert identity.name and identity.revision and identity.source

    def test_provenance_is_serialisable(self, key: str) -> None:
        provider = _new_instance(key)
        dumped = provider.provenance().model_dump_json()
        assert isinstance(dumped, str) and len(dumped) > 0
