"""Tokenizer downloads are pinned to a Hub commit (M11 security review, finding
S-3): a tokenizer loaded by model name alone follows the repo's moving default
branch. The default model is always pinned; a caller's own model is pinned when
they pass ``revision=`` and warned about when they do not. A fake ``transformers``
module is used so nothing is downloaded."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from openbtk.data.clinical_text import tokenization


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return list(range(len(text.split())))


@pytest.fixture
def fake_transformers(monkeypatch: pytest.MonkeyPatch) -> Any:
    module = MagicMock()
    module.AutoTokenizer.from_pretrained = MagicMock(return_value=_FakeTokenizer())
    monkeypatch.setattr(tokenization, "require", lambda name, extra: module)
    monkeypatch.setattr(tokenization, "_tokenizer_cache", {})
    monkeypatch.setattr(tokenization, "_warned_unpinned", set())
    monkeypatch.setattr(tokenization, "log", MagicMock())
    return module


def _calls(module: Any) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    return [
        (c.args, c.kwargs) for c in module.AutoTokenizer.from_pretrained.call_args_list
    ]


def test_the_default_model_is_loaded_at_its_pinned_commit(
    fake_transformers: Any,
) -> None:
    assert tokenization.count_tokens_exact("a b c") == 3
    ((args, kwargs),) = _calls(fake_transformers)
    assert args == (tokenization.DEFAULT_EXACT_MODEL,)
    assert kwargs == {"revision": tokenization.DEFAULT_EXACT_REVISION}
    tokenization.log.warning.assert_not_called()


def test_a_callers_model_with_a_revision_is_pinned_and_not_warned_about(
    fake_transformers: Any,
) -> None:
    tokenization.count_tokens_exact("a b", model_name="org/model", revision="c0ffee")
    ((args, kwargs),) = _calls(fake_transformers)
    assert args == ("org/model",) and kwargs == {"revision": "c0ffee"}
    tokenization.log.warning.assert_not_called()


def test_a_callers_model_without_a_revision_is_loaded_unpinned_with_a_warning(
    fake_transformers: Any,
) -> None:
    tokenization.count_tokens_exact("a b", model_name="org/model")
    ((args, kwargs),) = _calls(fake_transformers)
    assert args == ("org/model",) and kwargs == {}
    event = tokenization.log.warning.call_args
    assert (
        event.args == ("tokenizer.unpinned",) and event.kwargs["model"] == "org/model"
    )


def test_the_unpinned_warning_is_once_per_model(fake_transformers: Any) -> None:
    for text in ("a", "b c"):
        tokenization._tokenizer_cache.clear()  # force a second load
        tokenization.count_tokens_exact(text, model_name="org/model")
    assert tokenization.log.warning.call_count == 1


def test_different_revisions_of_one_model_are_cached_separately(
    fake_transformers: Any,
) -> None:
    tokenization.count_tokens_exact("a", model_name="org/m", revision="aaa")
    tokenization.count_tokens_exact("a", model_name="org/m", revision="bbb")
    tokenization.count_tokens_exact("a", model_name="org/m", revision="aaa")
    assert [k["revision"] for _, k in _calls(fake_transformers)] == ["aaa", "bbb"]


def test_the_default_model_keeps_its_plain_cache_key(fake_transformers: Any) -> None:
    tokenization.count_tokens_exact("a")
    assert list(tokenization._tokenizer_cache) == [tokenization.DEFAULT_EXACT_MODEL]
