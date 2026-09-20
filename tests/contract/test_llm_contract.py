"""Shared contract every registered BaseLLMProvider must satisfy.

Parametrized over LLM_REGISTRY.list_keys() -- registering a provider
anywhere automatically enrols it here, with no opt-out (the same
no-exemptions policy as every other contract suite, docs/09_CODING_STANDARDS.md
section 12). Task 5.2 added the first four real providers; three of the
five checks below (sends_data_offsite, model_identity, provenance) never
touch the network or a model and so run unconditionally, but
generate/chat/stream make a REAL network call (OpenAI, Anthropic,
OpenAICompatibleProvider) or download and run a REAL model
(HuggingFaceLocalProvider) for every one of the four -- there is no
lightweight reference LLM to substitute the way tests/contract/conftest.py
provides one for loaders/chunkers/preprocessors, since "generate text from
a real model" has no meaningful zero-dependency stand-in. Those three
tests are skipped per key unless explicitly opted into
(OPENBTK_SLOW_TESTS=1, mirroring the project's existing
model-download/network gate) AND the actual resource each one needs is
genuinely available: a real API key for OpenAI/Anthropic (without one the
call would just fail with a predictable AuthenticationError, which is not
what this suite exists to verify), ``torch``+``transformers`` actually
installed for HuggingFaceLocalProvider (reproduced directly: this repo's
own dev venv has ``transformers`` but not ``torch``, and
OPENBTK_SLOW_TESTS=1 alone made this suite try, and fail, to load a real
model with the tensor backend missing), and a real reachable endpoint URL
for OpenAICompatibleProvider (there is no such server this repo controls
or can assume exists in any given environment, local or CI, so this one
needs an explicit endpoint named via an env var, not just a flag). Each
provider's own dedicated unit test file (tests/unit/llms/test_*.py)
covers generate/chat/stream/error-translation behaviour completely via
mocking, with 100% coverage, and runs unconditionally in every CI job.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from openbtk.core.config import PolicyConfig
from openbtk.core.registry import LLM_REGISTRY
from openbtk.core.schemas import LLMResponse, Message

from .conftest import enrolled

if TYPE_CHECKING:
    from openbtk.core.base import BaseLLMProvider

# This suite exists to verify each provider's own behaviour, not to
# re-verify sends_data_offsite/PolicyError enforcement itself -- that has
# its own dedicated suite (tests/security/test_offsite_policy_enforcement.py).
# Three of the four providers registered here send data offsite by design
# (OpenAIProvider, AnthropicProvider, OpenAICompatibleProvider), so every
# _new_instance() call needs the policy that permits constructing them, or
# Registry.create's own safe-by-default gate (checked before construction)
# would refuse every one of them before this suite ever got to test them.
_ALLOW_OFFSITE = PolicyConfig(allow_offsite_providers=True)

# Constructor kwargs for keys whose provider has required arguments beyond
# the reference-shape zero-arg case (the same per-key dispatch pattern
# already used by test_loader_contract.py's _make_source and
# test_chunker_contract.py's _make_record). openai_compatible's base_url
# is a harmless placeholder for the three tests that never make a real
# call (declares_sends_data_offsite/model_identity/provenance); the real,
# env-provided endpoint is substituted in _new_instance for the three that do.
_TINY_GPT2_SHA = "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"  # pragma: allowlist secret
_OPENAI_COMPATIBLE_BASE_URL_ENV_VAR = "OPENBTK_TEST_OPENAI_COMPATIBLE_BASE_URL"
_CONSTRUCTOR_KWARGS_BY_KEY: dict[str, dict[str, Any]] = {
    "llm.general.azure_openai": {"model": "a-deployment"},
    "llm.general.bedrock": {"model": "vendor.model-v1:0"},
    "llm.general.vertex": {"model": "model-001"},
    "llm.general.huggingface_local": {
        "model": "sshleifer/tiny-gpt2",
        "revision": _TINY_GPT2_SHA,
    },
    "llm.general.openai_compatible": {
        "base_url": "http://localhost:8000/v1",
        "model": "local-model",
    },
}

# Real-call requirement per key: an env var name means it must be set
# before attempting the call (a credential for the two cloud providers, a
# reachable endpoint URL for the compatible one); None means
# OPENBTK_SLOW_TESTS=1 is the only *env var* gate, but huggingface_local
# additionally needs its real dependencies actually installed, checked
# separately below.
_REAL_CALL_ENV_VAR_BY_KEY: dict[str, str | None] = {
    "llm.general.openai": "OPENAI_API_KEY",
    "llm.general.anthropic": "ANTHROPIC_API_KEY",
    "llm.general.azure_openai": "AZURE_OPENAI_API_KEY",
    "llm.general.bedrock": "AWS_ACCESS_KEY_ID",
    "llm.general.vertex": "GOOGLE_APPLICATION_CREDENTIALS",
    "llm.general.openai_compatible": _OPENAI_COMPATIBLE_BASE_URL_ENV_VAR,
    "llm.general.huggingface_local": None,
}

_HUGGINGFACE_LOCAL_KEY = "llm.general.huggingface_local"
_MISSING_DEPENDENCY_FOR_HUGGINGFACE_LOCAL: str | None = None
if importlib.util.find_spec("torch") is None:
    _MISSING_DEPENDENCY_FOR_HUGGINGFACE_LOCAL = "torch"
elif importlib.util.find_spec("transformers") is None:
    _MISSING_DEPENDENCY_FOR_HUGGINGFACE_LOCAL = "transformers"


def _skip_if_real_call_unavailable(key: str) -> None:
    if key not in _REAL_CALL_ENV_VAR_BY_KEY:
        return
    if os.environ.get("OPENBTK_SLOW_TESTS") != "1":
        pytest.skip(f"{key}: makes a real network/model call; set OPENBTK_SLOW_TESTS=1")
    required_env_var = _REAL_CALL_ENV_VAR_BY_KEY[key]
    if required_env_var is not None and not os.environ.get(required_env_var):
        pytest.skip(f"{key}: requires {required_env_var} for a real call")
    if key == _HUGGINGFACE_LOCAL_KEY and _MISSING_DEPENDENCY_FOR_HUGGINGFACE_LOCAL:
        pytest.skip(
            f"{key}: requires the 'llms' extra "
            f"({_MISSING_DEPENDENCY_FOR_HUGGINGFACE_LOCAL})"
        )


def _constructor_kwargs(key: str) -> dict[str, Any]:
    kwargs = dict(_CONSTRUCTOR_KWARGS_BY_KEY.get(key, {}))
    if key == "llm.general.openai_compatible":
        real_base_url = os.environ.get(_OPENAI_COMPATIBLE_BASE_URL_ENV_VAR)
        if real_base_url:
            kwargs["base_url"] = real_base_url
    return kwargs


def _new_instance(key: str) -> BaseLLMProvider:
    return LLM_REGISTRY.create(key, policy=_ALLOW_OFFSITE, **_constructor_kwargs(key))


class _Answer(BaseModel):
    value: str


@pytest.mark.parametrize("key", enrolled(LLM_REGISTRY))
class TestLLMContract:
    def test_declares_sends_data_offsite(self, key: str) -> None:
        provider = _new_instance(key)
        assert isinstance(provider.sends_data_offsite, bool)

    def test_generate_returns_llm_response(self, key: str) -> None:
        _skip_if_real_call_unavailable(key)
        provider = _new_instance(key)
        result = provider.generate("hello")
        assert isinstance(result, LLMResponse)
        assert isinstance(result.text, str)

    def test_stream_returns_iterator_of_strings(self, key: str) -> None:
        _skip_if_real_call_unavailable(key)
        provider = _new_instance(key)
        result = provider.stream("hello")
        assert isinstance(result, Iterator)
        chunks = list(result)
        assert all(isinstance(c, str) for c in chunks)

    def test_chat_returns_llm_response(self, key: str) -> None:
        _skip_if_real_call_unavailable(key)
        provider = _new_instance(key)
        result = provider.chat([Message(role="user", content="hi")])
        assert isinstance(result, LLMResponse)

    def test_model_identity_default_raises_unless_overridden(self, key: str) -> None:
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


def test_structured_default_parses_json_response() -> None:
    """The default structured() implementation (chat() + JSON validation)
    is only meaningfully testable against a provider whose chat() actually
    returns valid JSON for a given prompt -- the reference LLM's echo
    behaviour doesn't produce JSON, so this is tested directly rather than
    parametrized across every future provider, whose chat() responses are
    entirely provider-specific and cannot be assumed to be JSON at all."""

    class _JsonEchoLLM:
        def chat(self, messages: list[Message], **kwargs: object) -> LLMResponse:
            return LLMResponse(text='{"value": "ok"}')

    from openbtk.core.base import BaseLLMProvider

    result = BaseLLMProvider.structured(
        _JsonEchoLLM(),  # type: ignore[arg-type]
        [Message(role="user", content="respond in json")],
        _Answer,
    )
    assert isinstance(result, _Answer)
    assert result.value == "ok"
