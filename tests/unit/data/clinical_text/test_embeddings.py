"""Unit tests for opentbtk.data.clinical_text.embeddings.

These tests mock the underlying transformers model/tokenizer to avoid
downloading large model weights in CI. Shape and interface contracts are
verified; actual embedding quality is validated via integration/eval suites.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from opentbtk.data.clinical_text.embeddings import PubMedBERTEmbedding
from opentbtk.core.errors import EmbeddingError


class TestPubMedBERTEmbedding:
    def test_embed_empty_list_raises(self) -> None:
        embedder = PubMedBERTEmbedding()
        with pytest.raises(EmbeddingError, match="empty texts"):
            embedder.embed([])

    def test_dimension_default(self) -> None:
        embedder = PubMedBERTEmbedding()
        # dimension triggers _ensure_loaded if not cached; mock to avoid load
        embedder._dim = 768
        assert embedder.dimension == 768

    @patch("torch.no_grad")
    def test_embed_returns_correct_shape(self, mock_no_grad: MagicMock) -> None:
        embedder = PubMedBERTEmbedding(batch_size=2)

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_encoded = MagicMock()
        mock_encoded.to.return_value = mock_encoded
        mock_attention_mask = MagicMock()
        mock_attention_mask.unsqueeze.return_value.float.return_value = (
            __import__("torch").ones(2, 5, 1)
        )
        mock_encoded.__getitem__.side_effect = lambda k: (
            mock_attention_mask if k == "attention_mask" else MagicMock()
        )
        mock_tokenizer.return_value = mock_encoded

        # Mock model output
        import torch
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(2, 5, 768)
        mock_model = MagicMock(return_value=mock_output)
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None
        mock_model.config.hidden_size = 768

        embedder._tokenizer = mock_tokenizer
        embedder._model = mock_model
        embedder._dim = 768

        result = embedder.embed(["text one", "text two"])
        assert result.shape == (2, 768)
        assert result.dtype == np.float32

    def test_missing_transformers_raises_clear_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins
        real_import = builtins.__import__

        def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "transformers":
                raise ImportError("mocked missing transformers")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        embedder = PubMedBERTEmbedding()
        with pytest.raises(EmbeddingError, match="transformers"):
            embedder.embed(["some text"])
