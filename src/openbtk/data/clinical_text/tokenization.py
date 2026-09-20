"""Subword token counting (docs/05_DATA_MODALITY_SPEC.md section 1.3).

Whitespace counting undercounts clinical text by roughly 30% (abbreviations,
drug names, measurements), so a chunker trusting an approximate count can
silently produce chunks that overflow the target model's context window.
Real counting needs the tokenizer of the target embedding model, via
HuggingFace ``transformers``, lazily loaded -- never in ``__init__``
(CLAUDE.md rule 11), only inside ``count_tokens_exact`` on first use.

**Both modes are fully real; which one is a caller's *default* is a
separate decision.** The spec's own wording -- "the approximation is
available but must be requested explicitly" -- states an accuracy
preference for exact counting, not a testing-infrastructure one.
``SectionAwareChunker`` (task 3.5) is where a concrete default gets
chosen, and it deliberately does NOT default to exact counting: this
project's own established precedent (``DeidEngine``'s default recognizer
set, ``NERRecognizer`` being opt-in) is that nothing needs a network
download and an extra installed just by being *constructed and used with
defaults*. This is a disclosed deviation from the spec's literal
preference, not a silent one -- see ``SectionAwareChunker``'s own
docstring for exactly how it resolves this.
"""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING

from openbtk.core._lazy import require
from openbtk.core.logging import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

DEFAULT_EXACT_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
"""The tokenizer used when a caller does not name one -- PubMedBERT-family,
matching docs/03_ARCHITECTURE.md section 6.1's named biomedical embedding
models. A future embedding provider (M5+) may want ITS OWN tokenizer
instead; `count_tokens_exact(text, model_name=...)` takes any HF model id."""

DEFAULT_EXACT_REVISION = (
    "e1354b7a3a09615f6aba48dfad4b7a613eef7062"  # pragma: allowlist secret
)
"""The Hub commit `DEFAULT_EXACT_MODEL` is pinned to (verified against the Hub
when the embedding presets were written). A tokenizer is code *and* data
downloaded from a moving branch unless pinned, so the default is never unpinned."""

log = get_logger(__name__)

_tokenizer_cache: dict[str, PreTrainedTokenizerBase] = {}
_warned_unpinned: set[str] = set()

_WHITESPACE_RE = re.compile(r"\S+")


def _get_tokenizer(
    model_name: str, revision: str | None = None
) -> PreTrainedTokenizerBase:
    if revision is None and model_name == DEFAULT_EXACT_MODEL:
        revision = DEFAULT_EXACT_REVISION
    cache_key = (
        model_name
        if revision is None or revision == DEFAULT_EXACT_REVISION
        else f"{model_name}@{revision}"
    )
    if cache_key not in _tokenizer_cache:
        transformers = require("transformers", extra="text")
        kwargs: dict[str, str] = {}
        if revision is not None:
            kwargs["revision"] = revision
        elif model_name not in _warned_unpinned:
            _warned_unpinned.add(model_name)
            log.warning(
                "tokenizer.unpinned",
                model=model_name,
                message=(
                    "Loading a tokenizer without a pinned revision follows the "
                    "model repo's moving default branch; pass revision=<commit>."
                ),
            )
        # huggingface_hub emits a "cache-system uses symlinks" UserWarning
        # on a machine without symlink support (e.g. Windows without
        # Developer Mode) -- third-party environment noise, confirmed by
        # direct reproduction, unrelated to anything in this module.
        # openbtk's own pytest config promotes every warning to an error
        # (pyproject.toml's filterwarnings); scoped narrowly here rather
        # than weakening that policy project-wide.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*uses symlinks.*")
            # bandit B615 (unpinned Hub download) is suppressed on the call below: the
            # default model is pinned, and a caller's own model without a revision
            # is warned about (finding S-3).
            _tokenizer_cache[cache_key] = transformers.AutoTokenizer.from_pretrained(  # nosec B615
                model_name, **kwargs
            )
    return _tokenizer_cache[cache_key]


def count_tokens_exact(
    text: str,
    *,
    model_name: str = DEFAULT_EXACT_MODEL,
    revision: str | None = None,
) -> int:
    """Exact subword count using ``model_name``'s real tokenizer.

    Args:
        text: The text to count. Never logged.
        model_name: A HuggingFace model id. Its tokenizer is downloaded
            and cached on first use for that id.
        revision: The Hub commit to load. The default model is always pinned;
            for any other model, omitting this follows the repo's moving
            default branch and logs a ``tokenizer.unpinned`` warning.

    Raises:
        MissingDependencyError: If the ``text`` extra is not installed.
    """
    if not text:
        return 0
    tokenizer = _get_tokenizer(model_name, revision)
    return len(tokenizer.encode(text, add_special_tokens=False))


def count_tokens_approximate(text: str) -> int:
    """Whitespace-based approximation: number of non-whitespace runs.

    A documented undercount relative to a real subword tokenizer (roughly
    30% on clinical text, per docs/05_DATA_MODALITY_SPEC.md section 1.3)
    -- exists as the zero-dependency fallback, never a silent claim of
    exactness.
    """
    return len(_WHITESPACE_RE.findall(text))
