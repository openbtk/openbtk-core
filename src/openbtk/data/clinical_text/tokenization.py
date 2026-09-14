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

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

DEFAULT_EXACT_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
"""The tokenizer used when a caller does not name one -- PubMedBERT-family,
matching docs/03_ARCHITECTURE.md section 6.1's named biomedical embedding
models. A future embedding provider (M5+) may want ITS OWN tokenizer
instead; `count_tokens_exact(text, model_name=...)` takes any HF model id."""

_tokenizer_cache: dict[str, PreTrainedTokenizerBase] = {}

_WHITESPACE_RE = re.compile(r"\S+")


def _get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    if model_name not in _tokenizer_cache:
        transformers = require("transformers", extra="text")
        # huggingface_hub emits a "cache-system uses symlinks" UserWarning
        # on a machine without symlink support (e.g. Windows without
        # Developer Mode) -- third-party environment noise, confirmed by
        # direct reproduction, unrelated to anything in this module.
        # openbtk's own pytest config promotes every warning to an error
        # (pyproject.toml's filterwarnings); scoped narrowly here rather
        # than weakening that policy project-wide.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*uses symlinks.*")
            _tokenizer_cache[model_name] = transformers.AutoTokenizer.from_pretrained(
                model_name
            )
    return _tokenizer_cache[model_name]


def count_tokens_exact(text: str, *, model_name: str = DEFAULT_EXACT_MODEL) -> int:
    """Exact subword count using ``model_name``'s real tokenizer.

    Args:
        text: The text to count. Never logged.
        model_name: A HuggingFace model id. Its tokenizer is downloaded
            and cached on first use for that id.

    Raises:
        MissingDependencyError: If the ``text`` extra is not installed.
    """
    if not text:
        return 0
    tokenizer = _get_tokenizer(model_name)
    return len(tokenizer.encode(text, add_special_tokens=False))


def count_tokens_approximate(text: str) -> int:
    """Whitespace-based approximation: number of non-whitespace runs.

    A documented undercount relative to a real subword tokenizer (roughly
    30% on clinical text, per docs/05_DATA_MODALITY_SPEC.md section 1.3)
    -- exists as the zero-dependency fallback, never a silent claim of
    exactness.
    """
    return len(_WHITESPACE_RE.findall(text))
