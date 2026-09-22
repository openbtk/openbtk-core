"""Shared contract every registered BaseEmbeddingProvider must satisfy.

Parametrized over EMBEDDING_REGISTRY.list_keys() -- no opt-out
(docs/09_CODING_STANDARDS.md section 12). Task 5.4 added the first two
real providers; three of the five checks
(sends_data_offsite/dimension/model_identity/provenance -- four, not
three; dimension is required to answer without a real model load, see
HuggingFaceEmbeddingProvider's own docstring) run unconditionally, since
none of them touch the network or a model. ``embed``/``embed_one`` make a
REAL network call (OpenAI) or download and run a REAL model
(HuggingFaceEmbeddingProvider) -- skipped per key unless
OPENBTK_SLOW_TESTS=1 AND the exact resource each one needs is genuinely
available, the identical reasoning and gating shape as
tests/contract/test_llm_contract.py (see that file's own docstring for
why "requires torch+transformers" and "requires an API key" are checked
separately from the opt-in flag itself). Each provider's own dedicated
unit test file (tests/unit/embeddings/test_*.py) covers
embed/error-translation/pooling behaviour completely via mocking, with
100% coverage, and runs unconditionally in every CI job.
"""

from __future__ import annotations

import importlib.util
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from openbtk.core.config import PolicyConfig
from openbtk.core.registry import EMBEDDING_REGISTRY

from .conftest import enrolled

if TYPE_CHECKING:
    from openbtk.core.base import BaseEmbeddingProvider

# See tests/contract/test_llm_contract.py's own docstring/comments for the
# rationale behind every piece of this gating shape -- it is deliberately
# identical here, not coincidentally similar.
_ALLOW_OFFSITE = PolicyConfig(allow_offsite_providers=True)

# A genuinely tiny (126K-parameter) real BERT checkpoint published
# specifically for tests like this one -- not a real biomedical preset's
# multi-GB model, which would make an opt-in contract-suite run download
# gigabytes just to check a dimension/shape. Verified via the HuggingFace
# Hub API the same way every preset's own model/revision was.
_TINY_BERT_SHA = "f171d7baecaf37b5da5a3616d8833b9969753535"  # pragma: allowlist secret
_CONSTRUCTOR_KWARGS_BY_KEY: dict[str, dict[str, Any]] = {
    "embedding.general.huggingface": {
        "model": "hf-internal-testing/tiny-random-bert",
        "revision": _TINY_BERT_SHA,
        "dimension": 32,
    },
}

_HUGGINGFACE_LOCAL_KEY = "embedding.general.huggingface"
_REAL_CALL_ENV_VAR_BY_KEY: dict[str, str | None] = {
    "embedding.general.openai": "OPENAI_API_KEY",
    _HUGGINGFACE_LOCAL_KEY: None,
}

_MISSING_DEPENDENCY_FOR_HUGGINGFACE: str | None = None
if importlib.util.find_spec("torch") is None:
    _MISSING_DEPENDENCY_FOR_HUGGINGFACE = "torch"
elif importlib.util.find_spec("transformers") is None:
    _MISSING_DEPENDENCY_FOR_HUGGINGFACE = "transformers"


def _skip_if_real_call_unavailable(key: str) -> None:
    if key not in _REAL_CALL_ENV_VAR_BY_KEY:
        return
    if os.environ.get("OPENBTK_SLOW_TESTS") != "1":
        pytest.skip(f"{key}: makes a real network/model call; set OPENBTK_SLOW_TESTS=1")
    required_env_var = _REAL_CALL_ENV_VAR_BY_KEY[key]
    if required_env_var is not None and not os.environ.get(required_env_var):
        pytest.skip(f"{key}: requires {required_env_var} for a real call")
    if key == _HUGGINGFACE_LOCAL_KEY and _MISSING_DEPENDENCY_FOR_HUGGINGFACE:
        pytest.skip(
            f"{key}: requires the 'llms' extra ({_MISSING_DEPENDENCY_FOR_HUGGINGFACE})"
        )


def _new_instance(key: str) -> BaseEmbeddingProvider:
    return EMBEDDING_REGISTRY.create(
        key, policy=_ALLOW_OFFSITE, **_CONSTRUCTOR_KWARGS_BY_KEY.get(key, {})
    )


@pytest.mark.parametrize("key", enrolled(EMBEDDING_REGISTRY))
class TestEmbeddingContract:
    def test_declares_sends_data_offsite(self, key: str) -> None:
        """Every provider must declare this -- it is what FR-V-07's policy
        enforcement (allow_offsite_providers) reads to decide whether
        construction is even permitted under a local-only policy."""
        provider = _new_instance(key)
        assert isinstance(provider.sends_data_offsite, bool)

    def test_embed_shape_and_dtype(self, key: str) -> None:
        _skip_if_real_call_unavailable(key)
        provider = _new_instance(key)
        texts = ["one", "two", "three"]
        vectors = provider.embed(texts)
        assert vectors.shape == (len(texts), provider.dimension)
        assert vectors.dtype == np.float32

    def test_embed_one_matches_first_row_of_embed(self, key: str) -> None:
        _skip_if_real_call_unavailable(key)
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
