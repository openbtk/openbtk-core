"""
Biomedical text embedding providers.

Wraps HuggingFace `transformers` models for PubMedBERT, BioBERT,
ClinicalBERT, and SapBERT (entity-level embeddings for linking). All
providers conform to BaseEmbeddingProvider and are registered under the
EMBEDDING_REGISTRY for config-driven instantiation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from opentbtk.core.base import BaseEmbeddingProvider
from opentbtk.core.errors import EmbeddingError
from opentbtk.core.registry import EMBEDDING_REGISTRY

log = structlog.get_logger(__name__)


class _TransformersEmbeddingBase(BaseEmbeddingProvider):
    """Shared implementation for HuggingFace encoder-based embedding providers.

    Subclasses set `_default_model_id` and `_default_dimension`. Uses
    mean-pooling over the last hidden state (standard for BERT-family
    sentence embeddings without a dedicated pooling head).
    """

    _default_model_id: str = ""
    _default_dimension: int = 768

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self._model_id = model_name or self._default_model_id
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._model: Any = None
        self._tokenizer: Any = None
        self._dim: int | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise EmbeddingError(
                "transformers and torch are required for biomedical text "
                "embeddings. Install with: pip install opentbtk[clinical_text]",
            ) from e

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
            self._model = AutoModel.from_pretrained(self._model_id)
            self._model.to(self._device)
            self._model.eval()
        except Exception as e:
            raise EmbeddingError(
                f"Failed to load embedding model '{self._model_id}'",
                context={"model_id": self._model_id},
            ) from e

        self._dim = self._model.config.hidden_size
        log.info(
            "embedding.loaded",
            model_id=self._model_id,
            device=self._device,
            dimension=self._dim,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using mean-pooled last hidden states.

        Args:
            texts: List of input strings. Must be non-empty.

        Returns:
            Array of shape (len(texts), dimension), dtype float32.

        Raises:
            EmbeddingError: On model load or inference failure, or if
                `texts` is empty.
        """
        if not texts:
            raise EmbeddingError("embed() called with empty texts list")
        self._ensure_loaded()

        import torch

        all_vectors: list[np.ndarray] = []
        try:
            with torch.no_grad():
                for i in range(0, len(texts), self._batch_size):
                    batch = texts[i : i + self._batch_size]
                    encoded = self._tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=self._max_length,
                        return_tensors="pt",
                    ).to(self._device)

                    output = self._model(**encoded)
                    last_hidden = output.last_hidden_state  # (B, T, H)

                    mask = encoded["attention_mask"].unsqueeze(-1).float()
                    summed = (last_hidden * mask).sum(dim=1)
                    counts = mask.sum(dim=1).clamp(min=1e-9)
                    mean_pooled = summed / counts

                    all_vectors.append(mean_pooled.cpu().numpy().astype(np.float32))
        except Exception as e:
            raise EmbeddingError(
                "Embedding inference failed",
                context={"model_id": self._model_id, "n_texts": len(texts)},
            ) from e

        return np.concatenate(all_vectors, axis=0)

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._ensure_loaded()
        assert self._dim is not None
        return self._dim


@EMBEDDING_REGISTRY.register("embedding.clinical_text.pubmedbert")
class PubMedBERTEmbedding(_TransformersEmbeddingBase):
    """Embeddings via Microsoft PubMedBERT (trained on PubMed abstracts + full text).

    Args:
        model_name: Override the default model ID if desired.
        device: "cpu", "cuda", or "mps".
        batch_size: Texts per inference batch.
        max_length: Max token length before truncation.
    """
    _default_model_id = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    _default_dimension = 768


@EMBEDDING_REGISTRY.register("embedding.clinical_text.biobertv1")
class BioBERTEmbedding(_TransformersEmbeddingBase):
    """Embeddings via BioBERT v1.2 (trained on PubMed abstracts + PMC full text)."""
    _default_model_id = "dmis-lab/biobert-base-cased-v1.2"
    _default_dimension = 768


@EMBEDDING_REGISTRY.register("embedding.clinical_text.clinicalbert")
class ClinicalBERTEmbedding(_TransformersEmbeddingBase):
    """Embeddings via Bio_ClinicalBERT (BioBERT further pretrained on MIMIC-III notes)."""
    _default_model_id = "emilyalsentzer/Bio_ClinicalBERT"
    _default_dimension = 768


@EMBEDDING_REGISTRY.register("embedding.clinical_text.sapbert")
class SapBERTEmbedding(_TransformersEmbeddingBase):
    """Entity-level embeddings via SapBERT, optimized for biomedical entity linking.

    SapBERT is self-aligned on UMLS synonym pairs, producing embeddings well
    suited to nearest-neighbor entity normalization (e.g., mapping a surface
    form like "heart attack" close to "myocardial infarction"). Prefer this
    over PubMedBERT/BioBERT when embedding short entity mentions rather than
    full sentences/passages.
    """
    _default_model_id = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    _default_dimension = 768
