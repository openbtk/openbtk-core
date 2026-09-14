"""Release-relevant gap, tracked honestly rather than hidden: T3 in
docs/06_SECURITY_COMPLIANCE.md section 1 ("PHI sent to a third-party API")
is supposed to be closed by policy enforcement described in section 3.3 and
in core/base.py's own module docstring:

    **``sends_data_offsite`` is enforced, not advisory.** Under
    ``policy.allow_offsite_providers: false`` (the pipeline default),
    constructing a provider with this set to ``True`` raises
    ``PolicyError``. That enforcement lives in the pipeline/registry layer,
    not here; this module only declares the contract.

As of M1, that enforcement does not exist anywhere. ``sends_data_offsite``
(base.py), ``PolicyConfig.allow_offsite_providers`` (config.py), and
``PolicyError`` (errors.py) are three declared, disconnected pieces --
confirmed by grep, not by reading the docstring's claim at face value.
``Registry.create`` (registry.py) takes no policy argument at all and
performs no such check. This is honestly attributable: base.py's own
docstring says the enforcement "lives in the pipeline/registry layer", and
no pipeline layer exists yet (it is M3+ on the roadmap) -- so this is
exactly the kind of gap that is expected to still be open at M1, not a
regression.

This test drives the REAL registry construction path (not a local
reimplementation of the missing check -- that would trivially "pass" for
the wrong reason and defeat the point) and asserts the policy-safe outcome
that base.py's docstring promises. It is marked ``xfail(strict=True)`` so
that:
    - it fails LOUDLY for as long as the gap exists, rather than being
      silently skipped or, worse, quietly omitted from the suite entirely,
    - and the moment real enforcement is added anywhere in this path, this
      test starts unexpectedly PASSING, which ``strict=True`` turns into a
      hard failure -- forcing whoever implements enforcement to come here
      and turn this into a real regression test, rather than leaving a
      stale xfail as fake coverage forever.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pytest

from openbtk.core.base import BaseEmbeddingProvider
from openbtk.core.config import PolicyConfig
from openbtk.core.errors import PolicyError
from openbtk.core.registry import EMBEDDING_REGISTRY

if TYPE_CHECKING:
    from numpy.typing import NDArray

_OFFSITE_PROBE_KEY = "embedding.general.offsite_policy_probe"

if not EMBEDDING_REGISTRY.is_registered(_OFFSITE_PROBE_KEY):

    @EMBEDDING_REGISTRY.register(_OFFSITE_PROBE_KEY)
    class _FakeOffsiteEmbeddingProvider(BaseEmbeddingProvider):
        """A real, registrable BaseEmbeddingProvider standing in for
        OpenAIEmbedding (docs/06_SECURITY_COMPLIANCE.md section 3.3's own
        example of a provider with sends_data_offsite=True). Registering
        into the real global EMBEDDING_REGISTRY means the embedding
        contract suite sweeps this up too (by design: no opt-out -- the
        same test-order-pollution lesson tests/unit/core/test_config.py's
        probe loader already ran into), so embed() must be a genuinely
        working implementation, not a stub -- constructing the instance,
        not calling embed(), is the operation that base.py's docstring
        says must be policy-gated."""

        sends_data_offsite: ClassVar[bool] = True

        def embed(self, texts: list[str]) -> NDArray[np.float32]:
            return np.zeros((len(texts), self.dimension), dtype=np.float32)

        @property
        def dimension(self) -> int:
            return 1


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known M1 gap, tracked deliberately: no enforcement mechanism "
        "connects sends_data_offsite, PolicyConfig.allow_offsite_providers, "
        "and PolicyError yet. base.py's own module docstring says this "
        "enforcement belongs in the pipeline/registry layer; "
        "openbtk.pipelines does not exist until M3+ on the roadmap, and "
        "Registry.create takes no policy argument today. When enforcement "
        "is implemented, this test will start passing unexpectedly "
        "(strict=True catches that) -- turn it into a real regression test "
        "against the actual enforcement API at that point."
    ),
)
def test_offsite_provider_blocked_by_default_policy() -> None:
    policy = PolicyConfig()  # allow_offsite_providers=False, the safe default
    assert policy.allow_offsite_providers is False

    # What base.py's docstring promises: constructing an offsite provider
    # under this policy raises PolicyError, before any data is read. Today,
    # Registry.create has no policy parameter and performs no such check --
    # so this construction silently succeeds, and this assertion is the
    # honest, precise shape of "not implemented yet".
    with pytest.raises(PolicyError):
        EMBEDDING_REGISTRY.create(_OFFSITE_PROBE_KEY)
