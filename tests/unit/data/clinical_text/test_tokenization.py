"""Unit tests for opentbtk.data.clinical_text.tokenization."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from opentbtk.data.clinical_text.tokenization import BiomedicalTokenizer
from opentbtk.core.errors import ProcessingError


class TestBiomedicalTokenizer:
    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(ProcessingError, match="Unknown tokenizer preset"):
            BiomedicalTokenizer(preset="not_a_real_preset")

    def test_preset_resolves_to_known_model_id(self) -> None:
        tok = BiomedicalTokenizer(preset="pubmedbert")
        assert "PubMedBERT" in tok.model_name

    def test_explicit_model_name_overrides_preset(self) -> None:
        tok = BiomedicalTokenizer(preset="pubmedbert", model_name="custom/model-id")
        assert tok.model_name == "custom/model-id"

    def test_tokenize_uses_mocked_hf_tokenizer(self) -> None:
        tok = BiomedicalTokenizer(preset="pubmedbert")
        mock_hf = MagicMock()
        mock_hf.tokenize.return_value = ["patient", "presents", "with", "pain"]
        tok._tokenizer = mock_hf

        result = tok.tokenize("Patient presents with pain.")
        assert result == ["patient", "presents", "with", "pain"]

    def test_count_tokens_uses_mocked_encode(self) -> None:
        tok = BiomedicalTokenizer(preset="pubmedbert")
        mock_hf = MagicMock()
        mock_hf.encode.return_value = [101, 102, 103]
        tok._tokenizer = mock_hf

        count = tok.count_tokens("some text")
        assert count == 3
        mock_hf.encode.assert_called_with("some text", add_special_tokens=False)

    def test_decode_uses_mocked_tokenizer(self) -> None:
        tok = BiomedicalTokenizer(preset="pubmedbert")
        mock_hf = MagicMock()
        mock_hf.decode.return_value = "decoded text"
        tok._tokenizer = mock_hf

        result = tok.decode([1, 2, 3])
        assert result == "decoded text"

    def test_vocab_size_property(self) -> None:
        tok = BiomedicalTokenizer(preset="pubmedbert")
        mock_hf = MagicMock()
        mock_hf.vocab_size = 30522
        tok._tokenizer = mock_hf

        assert tok.vocab_size == 30522
