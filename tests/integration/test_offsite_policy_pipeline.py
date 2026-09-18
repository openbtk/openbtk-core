"""``sends_data_offsite`` + ``PolicyError`` enforcement, end to end (task
5.5) -- through the real ``Pipeline``/executor and a REAL registered
offsite provider, closing the gap task 5.2's own roadmap note
anticipated: "this can only be REALLY tested once a real offsite provider
like OpenAI/Anthropic exists."

tests/security/test_offsite_policy_enforcement.py already proves the
*mechanism* (``Registry.create``/``create_from_config``) in isolation,
against fake test doubles on a private registry instance -- deliberately
so the embedding contract suite's real, no-opt-out sweep never trips over
a fake offsite provider planted in the real global registry. This file
proves the same enforcement actually fires when a real user builds a real
``Pipeline`` naming a real registered offsite provider
(``llm.general.openai``, ``embedding.general.openai``) -- no test double,
no private registry, no mocking of the SDK: construction is blocked (or
not) before ``require()`` is ever reached, so neither scenario below
needs the real ``openai`` package installed.

No network call happens either way: a blocked construction never gets
far enough to build the SDK client, and an *allowed* one still fails --
for a completely different, already-disclosed reason (the executor has
no ``llm``/``embedding``-category dispatch yet, per its own module
docstring) -- before any call would be made either. Proving those two
failure messages are genuinely different is exactly what demonstrates
the offsite gate is what changed, not that the run coincidentally failed
for the same reason twice.
"""

from __future__ import annotations

from openbtk import embeddings as _embeddings  # noqa: F401 -- registers embedding.*
from openbtk import llms as _llms  # noqa: F401 -- registers llm.*, collected standalone
from openbtk.core.config import PolicyConfig
from openbtk.pipelines import Pipeline, Step


class TestLLMStepOffsiteEnforcement:
    def test_blocked_by_default_with_no_policy_opt_in(self) -> None:
        pipeline = Pipeline("offsite-probe").add(Step("llm", "llm.general.openai"))
        manifest = pipeline.run()
        assert manifest.status == "failed"
        assert manifest.error is not None
        assert "sends data offsite" in manifest.error
        assert "OpenAIProvider" in manifest.error

    def test_construction_gets_past_the_policy_gate_once_allowed(self) -> None:
        pipeline = Pipeline(
            "offsite-probe", policy=PolicyConfig(allow_offsite_providers=True)
        ).add(Step("llm", "llm.general.openai"))
        manifest = pipeline.run()
        # Still "failed" -- but now for the executor's separate, disclosed
        # llm-dispatch gap, never reached while the offsite gate itself
        # was still the blocker. This is the real proof the two are
        # distinct failure modes, not the same block reported twice.
        assert manifest.status == "failed"
        assert manifest.error is not None
        assert "sends data offsite" not in manifest.error
        assert "not yet executable" in manifest.error


class TestEmbeddingStepOffsiteEnforcement:
    """Same mechanism, a different registry instance (EMBEDDING_REGISTRY,
    not LLM_REGISTRY) -- confirms enforcement isn't something that only
    happens to work for one specific registry."""

    def test_blocked_by_default_with_no_policy_opt_in(self) -> None:
        pipeline = Pipeline("offsite-probe").add(
            Step("embed", "embedding.general.openai")
        )
        manifest = pipeline.run()
        assert manifest.status == "failed"
        assert manifest.error is not None
        assert "sends data offsite" in manifest.error
        assert "OpenAIEmbeddingProvider" in manifest.error

    def test_construction_gets_past_the_policy_gate_once_allowed(self) -> None:
        pipeline = Pipeline(
            "offsite-probe", policy=PolicyConfig(allow_offsite_providers=True)
        ).add(Step("embed", "embedding.general.openai"))
        manifest = pipeline.run()
        assert manifest.status == "failed"
        assert manifest.error is not None
        assert "sends data offsite" not in manifest.error
        assert "not yet executable" in manifest.error


class TestLocalOnlyProviderIsUnaffectedThroughTheRealPipeline:
    """Negative control, through the real pipeline this time (the fake
    registry version already lives in test_offsite_policy_enforcement.py):
    a local, non-offsite provider must never trip this gate."""

    def test_a_local_llm_step_is_not_blocked_by_the_offsite_gate(self) -> None:
        sha = "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"  # pragma: allowlist secret
        pipeline = Pipeline("local-probe").add(
            Step(
                "llm",
                "llm.general.huggingface_local",
                model="sshleifer/tiny-gpt2",
                revision=sha,
            )
        )
        manifest = pipeline.run()
        # Reaches the same "not yet executable" failure as the allowed
        # offsite case above -- proving construction itself succeeded,
        # since sends_data_offsite=False never engages the policy gate at all.
        assert manifest.status == "failed"
        assert manifest.error is not None
        assert "sends data offsite" not in manifest.error
        assert "not yet executable" in manifest.error
