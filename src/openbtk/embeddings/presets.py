"""Biomedical embedding presets: configuration, not bespoke provider classes.

Same reasoning as :mod:`openbtk.llms.presets` (task 5.3), applied to task
5.4's five named models: PubMedBERT, BioBERT, ClinicalBERT, SapBERT and
MedCPT are all ordinary BERT-family encoders
:class:`~openbtk.embeddings.huggingface.HuggingFaceEmbeddingProvider`
already handles -- a preset pins the right ``model``/``revision``/
``dimension``/``pooling``, not a new class per model.

Every ``model``, ``revision`` and ``dimension`` below was verified
directly against the HuggingFace Hub (``GET /api/models/<id>`` for the
commit SHA, and that model's own ``config.json`` for ``hidden_size``) at
the time this module was written, not fabricated (CLAUDE.md rule 14).
Two things worth knowing about the values themselves:

* **PubMedBERT was renamed.** ``microsoft/BiomedNLP-PubMedBERT-base-
  uncased-abstract-fulltext`` (the name in most papers and this project's
  own roadmap) now 307-redirects on the Hub to
  ``microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`` --
  confirmed directly, not assumed. The preset is still named
  ``"pubmedbert"`` (the common name everyone actually searches for) but
  points at the real, current repository id.
* **MedCPT is a dual encoder**, not one symmetric model: NCBI publishes a
  separate Query-Encoder and Article-Encoder, each optimised for its own
  role rather than interchangeable. Presenting a single "medcpt" preset
  using just one of them would misrepresent how the model is meant to be
  used, so both are named explicitly (``medcpt-query`` for queries,
  ``medcpt-article`` for the documents/chunks a retrieval corpus embeds
  once and stores).

**Pooling:** ``"mean"`` for PubMedBERT/BioBERT/ClinicalBERT (plain MLM
checkpoints with no single documented pooling convention of their own --
masked mean is the general-purpose default); ``"cls"`` for SapBERT and
MedCPT, whose own papers and model cards specifically document CLS-token
pooling as their intended usage.

None of these five models are gated on the Hub (unlike some of the LLM
presets in :mod:`openbtk.llms.presets`) -- confirmed via the same API call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openbtk.core.errors import ConfigError
from openbtk.core.registry import EMBEDDING_REGISTRY

if TYPE_CHECKING:
    from openbtk.core.base import BaseEmbeddingProvider

_HUGGINGFACE_KEY = "embedding.general.huggingface"

# See this module's docstring: the redirect was confirmed directly.
_PUBMEDBERT_SHA = "e1354b7a3a09615f6aba48dfad4b7a613eef7062"  # pragma: allowlist secret
_BIOBERT_SHA = "67c9c25b46986521ca33df05d8540da1210b3256"  # pragma: allowlist secret
_CLINBERT_SHA = "d5892b39a4adaed74b92212a44081509db72f87b"  # pragma: allowlist secret
_SAPBERT_SHA = "090663c3ae57bf35ffe4d0d468a2a88d03051a4d"  # pragma: allowlist secret
_MEDCPT_Q_SHA = "d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc"  # pragma: allowlist secret
_MEDCPT_A_SHA = "d05a736da4bb84ee4057b7f7999485be6ed85465"  # pragma: allowlist secret
_SAPBERT_MODEL = (
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"  # pragma: allowlist secret
)

BIOMEDICAL_EMBEDDING_PRESETS: dict[str, dict[str, Any]] = {
    "pubmedbert": {
        "type": _HUGGINGFACE_KEY,
        "params": {
            "model": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
            "revision": _PUBMEDBERT_SHA,
            "dimension": 768,
            "pooling": "mean",
        },
    },
    "biobert": {
        "type": _HUGGINGFACE_KEY,
        "params": {
            "model": "dmis-lab/biobert-base-cased-v1.2",
            "revision": _BIOBERT_SHA,
            "dimension": 768,
            "pooling": "mean",
        },
    },
    "clinicalbert": {
        "type": _HUGGINGFACE_KEY,
        "params": {
            "model": "emilyalsentzer/Bio_ClinicalBERT",
            "revision": _CLINBERT_SHA,
            "dimension": 768,
            "pooling": "mean",
        },
    },
    "sapbert": {
        "type": _HUGGINGFACE_KEY,
        "params": {
            "model": _SAPBERT_MODEL,
            "revision": _SAPBERT_SHA,
            "dimension": 768,
            "pooling": "cls",
        },
    },
    "medcpt-query": {
        "type": _HUGGINGFACE_KEY,
        "params": {
            "model": "ncbi/MedCPT-Query-Encoder",
            "revision": _MEDCPT_Q_SHA,
            "dimension": 768,
            "pooling": "cls",
        },
    },
    "medcpt-article": {
        "type": _HUGGINGFACE_KEY,
        "params": {
            "model": "ncbi/MedCPT-Article-Encoder",
            "revision": _MEDCPT_A_SHA,
            "dimension": 768,
            "pooling": "cls",
        },
    },
}


def list_embedding_presets() -> list[str]:
    """Return every preset name, sorted."""
    return sorted(BIOMEDICAL_EMBEDDING_PRESETS)


def create_embedding_preset(name: str, /, **overrides: Any) -> BaseEmbeddingProvider:
    """Construct the provider a biomedical embedding preset names.

    Args:
        name: One of :func:`list_embedding_presets`'s names.
        **overrides: Merged over the preset's own params before
            construction -- e.g. ``device="cuda"`` or ``policy=...``.

    Returns:
        The constructed, registered provider (currently always a
        :class:`~openbtk.embeddings.huggingface.HuggingFaceEmbeddingProvider`).

    Raises:
        ConfigError: If ``name`` is not a known preset.

    Example:
        >>> sorted(list_embedding_presets())  # doctest: +NORMALIZE_WHITESPACE
        ['biobert', 'clinicalbert', 'medcpt-article', 'medcpt-query',
         'pubmedbert', 'sapbert']
    """
    preset = BIOMEDICAL_EMBEDDING_PRESETS.get(name)
    if preset is None:
        raise ConfigError(
            f"Unknown biomedical embedding preset {name!r}. "
            f"Available: {list_embedding_presets()}",
            context={"preset": name},
        )
    policy = overrides.pop("policy", None)
    config = {
        "type": preset["type"],
        "params": {**preset["params"], **overrides},
    }
    return EMBEDDING_REGISTRY.create_from_config(config, policy=policy)
