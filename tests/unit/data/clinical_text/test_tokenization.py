"""Unit tests for openbtk.data.clinical_text.tokenization.

Approximate mode is tested exhaustively (zero dependencies, always
available). Exact mode's real HF-tokenizer tests are ``@pytest.mark.slow``.
"""

from __future__ import annotations

import pytest

from openbtk.core.errors import MissingDependencyError
from openbtk.data.clinical_text import tokenization
from openbtk.data.clinical_text.tokenization import (
    count_tokens_approximate,
    count_tokens_exact,
)


class TestApproximate:
    def test_counts_whitespace_separated_runs(self) -> None:
        assert count_tokens_approximate("chest pain and fever") == 4

    def test_empty_text_is_zero(self) -> None:
        assert count_tokens_approximate("") == 0

    def test_multiple_spaces_do_not_inflate_the_count(self) -> None:
        assert count_tokens_approximate("chest    pain") == 2

    def test_a_single_long_word_is_one_approximate_token(self) -> None:
        """The mechanism the documented undercount comes from: whitespace
        splitting always counts one word as one token regardless of how
        many real subwords it is -- TestExactReal (slow) compares this
        against a real tokenizer for a word verified to actually split."""
        assert count_tokens_approximate("hydrochlorothiazide") == 1


class TestExactMissingDependency:
    def test_missing_transformers_raises_missing_dependency_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_require(module: str, *, extra: str) -> None:
            raise MissingDependencyError(
                f"'{module}' is required but not installed.",
                context={"module": module, "extra": extra},
            )

        monkeypatch.setattr(tokenization, "require", fake_require)
        tokenization._tokenizer_cache.clear()
        with pytest.raises(MissingDependencyError) as exc_info:
            count_tokens_exact("hello")
        assert exc_info.value.context["extra"] == "text"

    def test_empty_text_returns_zero_without_needing_a_tokenizer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Short-circuits before require() -- constructing/calling with
        empty text must never force a download."""

        def exploding_require(module: str, *, extra: str) -> None:
            raise AssertionError("require() should not be called for empty text")

        monkeypatch.setattr(tokenization, "require", exploding_require)
        assert count_tokens_exact("") == 0


@pytest.mark.slow
class TestExactReal:
    """Requires the real "text" extra (transformers) and a downloaded
    tokenizer -- see tests/conftest.py's OPENBTK_SLOW_TESTS gate."""

    def test_counts_more_tokens_than_the_approximation_for_a_real_word(self) -> None:
        """Not every long clinical word undercounts under THIS model --
        PubMedBERT's domain vocabulary has "exacerbation" as a single
        token, for instance (a real, checked finding, not assumed) --
        but a drug name like this one reliably splits into real subwords
        under it, verified directly."""
        text = "hydrochlorothiazide"
        exact = count_tokens_exact(text)
        approx = count_tokens_approximate(text)
        assert exact > approx

    def test_is_deterministic(self) -> None:
        text = "Patient presents with acute exacerbation of COPD."
        assert count_tokens_exact(text) == count_tokens_exact(text)

    def test_tokenizer_is_cached_across_calls(self) -> None:
        tokenization._tokenizer_cache.clear()
        count_tokens_exact("first call")
        assert tokenization.DEFAULT_EXACT_MODEL in tokenization._tokenizer_cache
        cached = tokenization._tokenizer_cache[tokenization.DEFAULT_EXACT_MODEL]
        count_tokens_exact("second call")
        assert tokenization._tokenizer_cache[tokenization.DEFAULT_EXACT_MODEL] is cached
