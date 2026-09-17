"""Biomedical LLM presets: configuration, not bespoke provider classes.

Task 5.3 is explicitly scoped this way in docs/10_ROADMAP.md ("config, not
classes"): MedGemma, Meditron and OpenBioLLM are all causal language
models compatible with :class:`~openbtk.llms.huggingface.HuggingFaceLocalProvider`
as-is -- they need no code of their own, only the right ``model``/``revision``
pinned. Writing a ``MeditronProvider(HuggingFaceLocalProvider)`` subclass
per model would be a real violation of docs/09_CODING_STANDARDS.md section
7 ("wrap, don't reinvent") applied to this project's own code, not just
third-party libraries.

Each preset's ``revision`` is a real commit SHA, fetched directly from the
HuggingFace Hub API (``GET /api/models/<id>``) at the time this module was
written -- not fabricated (CLAUDE.md rule 14: no claim without a
benchmark, extended here to "no identifier without a verified source").
Re-verify and update these if a preset ever needs to move to a newer
revision; they do not track a model's "latest" automatically by design
(FR-P-05, the same reason :class:`HuggingFaceLocalProvider` requires an
explicit revision at all).

**Gating:** ``meditron-7b`` and ``medgemma-27b-text`` are both licensed,
"gated" models on the Hub (confirmed via the same API call) -- using
either preset requires a HuggingFace account that has accepted the
model's license and an authenticated ``huggingface-cli login`` (or
``HF_TOKEN``) locally; the preset itself does not and cannot bypass that.
``openbiollm-8b`` is openly licensed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openbtk.core.errors import ConfigError
from openbtk.core.registry import LLM_REGISTRY

if TYPE_CHECKING:
    from openbtk.core.base import BaseLLMProvider

# Each entry is exactly the {"type": ..., "params": {...}} shape
# Registry.create_from_config already accepts -- a preset is data,
# reusable as-is in a PipelineConfig YAML Step, not a Python-only concept.
# Real commit SHAs (see this module's own docstring for how these were
# obtained), named as constants rather than inlined below purely to keep
# each under the project's 88-column limit alongside its allowlist pragma.
_MEDITRON_SHA = "d7d0a5ed929384a6b059ac74198cf1d71f44ba76"  # pragma: allowlist secret
_OPENBIOLLM_SHA = "70d6bb521cab6ca755b675ade38831eedf89d31c"  # pragma: allowlist secret
_MEDGEMMA_SHA = "5b667cf2ddcf064085bc90952edb35a0edbfb79c"  # pragma: allowlist secret

BIOMEDICAL_LLM_PRESETS: dict[str, dict[str, Any]] = {
    "meditron-7b": {
        "type": "llm.general.huggingface_local",
        "params": {"model": "epfl-llm/meditron-7b", "revision": _MEDITRON_SHA},
    },
    "openbiollm-8b": {
        "type": "llm.general.huggingface_local",
        "params": {
            "model": "aaditya/Llama3-OpenBioLLM-8B",
            "revision": _OPENBIOLLM_SHA,
        },
    },
    # google/medgemma-4b-it is a real MedGemma release too, but is
    # multimodal (image-text-to-text, confirmed via the Hub API) --
    # architecturally incompatible with HuggingFaceLocalProvider's
    # text-only generate()/chat(). medgemma-27b-text-it is Google's
    # text-only variant of the same family (also confirmed via the API).
    "medgemma-27b-text": {
        "type": "llm.general.huggingface_local",
        "params": {
            "model": "google/medgemma-27b-text-it",
            "revision": _MEDGEMMA_SHA,
        },
    },
}


def list_llm_presets() -> list[str]:
    """Return every preset name, sorted."""
    return sorted(BIOMEDICAL_LLM_PRESETS)


def create_llm_preset(name: str, /, **overrides: Any) -> BaseLLMProvider:
    """Construct the provider a biomedical LLM preset names.

    Args:
        name: One of :func:`list_llm_presets`'s names.
        **overrides: Merged over the preset's own params before
            construction -- e.g. ``device="cuda"`` or ``policy=...``
            (forwarded through to :meth:`Registry.create_from_config`,
            which forwards it to :meth:`Registry.create`).

    Returns:
        The constructed, registered provider (currently always a
        :class:`~openbtk.llms.huggingface.HuggingFaceLocalProvider`).

    Raises:
        ConfigError: If ``name`` is not a known preset.

    Example:
        >>> sorted(list_llm_presets())
        ['medgemma-27b-text', 'meditron-7b', 'openbiollm-8b']
    """
    preset = BIOMEDICAL_LLM_PRESETS.get(name)
    if preset is None:
        raise ConfigError(
            f"Unknown biomedical LLM preset {name!r}. Available: {list_llm_presets()}",
            context={"preset": name},
        )
    policy = overrides.pop("policy", None)
    config = {
        "type": preset["type"],
        "params": {**preset["params"], **overrides},
    }
    return LLM_REGISTRY.create_from_config(config, policy=policy)
