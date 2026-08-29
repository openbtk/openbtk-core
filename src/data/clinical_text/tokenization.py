"""
Biomedical tokenization.

Wraps HuggingFace `transformers` tokenizers for biomedical-pretrained models.
No custom tokenization logic is implemented — this is a thin, consistent
adapter so the rest of the framework never imports `transformers` directly.
"""
from __future__ import annotations

from typing import Any

import structlog

from opentbtk.core.errors import ProcessingError
from opentbtk.core.registry import Registry

log = structlog.get_logger(__name__)

# Dedicated registry for tokenizers (not in the core category list since
# tokenization is specific to text-bearing modalities)
TOKENIZER_REGISTRY: Registry["BaseTokenizer"] = Registry("tokenizer")


# Known biomedical tokenizer model IDs, exposed as named presets so users
# don't need to remember exact HuggingFace Hub paths.
_PRESET_MODELS: dict[str, str] = {
    "bert-base": "bert-base-uncased",
    "biobert": "dmis-lab/biobert-base-cased-v1.2",
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "gatortron": "UFNLP/gatortron-base",
    "biogpt": "microsoft/biogpt",
}


class BaseTokenizer:
    """Interface for biomedical tokenizer adapters."""

    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, token_ids: list[int]) -> str:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError


@TOKENIZER_REGISTRY.register("tokenizer.clinical_text.biomedical")
class BiomedicalTokenizer(BaseTokenizer):
    """Tokenizer adapter wrapping a HuggingFace `transformers` tokenizer.

    Use the `preset` argument for common biomedical models, or pass an
    explicit `model_name` for any HuggingFace Hub tokenizer / local path.

    Args:
        preset: One of "bert-base", "biobert", "pubmedbert", "clinicalbert",
            "gatortron", "biogpt". Ignored if `model_name` is provided.
        model_name: Explicit HuggingFace model ID or local path. Takes
            precedence over `preset` if both are given.
        max_length: Maximum sequence length for truncation (default: 512,
            matching standard BERT-family context windows).

    Example:
        >>> tok = BiomedicalTokenizer(preset="pubmedbert")
        >>> tok.count_tokens("Patient presents with acute myocardial infarction.")
        9
    """

    def __init__(
        self,
        preset: str = "pubmedbert",
        model_name: str | None = None,
        max_length: int = 512,
    ) -> None:
        if model_name is None:
            if preset not in _PRESET_MODELS:
                raise ProcessingError(
                    f"Unknown tokenizer preset '{preset}'. "
                    f"Available presets: {sorted(_PRESET_MODELS)}",
                )
            model_name = _PRESET_MODELS[preset]

        self._model_name = model_name
        self._max_length = max_length
        self._tokenizer: Any = None

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None:
            return
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ProcessingError(
                "transformers is required for BiomedicalTokenizer. "
                "Install with: pip install opentbtk[clinical_text]",
            ) from e
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        except Exception as e:
            raise ProcessingError(
                f"Failed to load tokenizer '{self._model_name}' from HuggingFace Hub.",
                context={"model_name": self._model_name},
            ) from e
        log.info("tokenizer.loaded", model_name=self._model_name)

    def tokenize(self, text: str) -> list[str]:
        """Split text into subword tokens.

        Args:
            text: Input text.

        Returns:
            List of subword token strings.
        """
        self._ensure_loaded()
        return self._tokenizer.tokenize(text)

    def count_tokens(self, text: str) -> int:
        """Count the number of subword tokens in text (no special tokens added).

        Args:
            text: Input text.

        Returns:
            Token count.
        """
        self._ensure_loaded()
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs, truncated to max_length.

        Args:
            text: Input text.

        Returns:
            List of integer token IDs (including special tokens).
        """
        self._ensure_loaded()
        return self._tokenizer.encode(
            text, truncation=True, max_length=self._max_length
        )

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back into text.

        Args:
            token_ids: List of integer token IDs.

        Returns:
            Decoded text string.
        """
        self._ensure_loaded()
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        """Vocabulary size of the underlying tokenizer."""
        self._ensure_loaded()
        return int(self._tokenizer.vocab_size)

    @property
    def model_name(self) -> str:
        """The HuggingFace model ID or path backing this tokenizer."""
        return self._model_name
