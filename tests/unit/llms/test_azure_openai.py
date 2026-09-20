"""``AzureOpenAIProvider`` (FR-V-01): the OpenAI provider on an Azure resource. The
inherited request/response/retry behaviour is tested in ``test_openai.py``; these tests
cover what differs: the client, the deployment name and the provenance."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from openbtk.core.errors import AuthenticationError, RateLimitError
from openbtk.core.registry import LLM_REGISTRY
from openbtk.llms import azure_openai
from openbtk.llms.azure_openai import AzureOpenAIProvider
from openbtk.llms.openai import OpenAIProvider


class _AuthError(Exception):
    pass


class _RateError(Exception):
    pass


class _APIError(Exception):
    pass


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> Any:
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="hello"))],
        usage=MagicMock(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    module = MagicMock()
    module.AzureOpenAI = MagicMock(return_value=client)
    module.OpenAI = MagicMock(side_effect=AssertionError("must build an Azure client"))
    module.RateLimitError = _RateError
    module.AuthenticationError = _AuthError
    module.APIError = _APIError
    fake = lambda name, extra: module  # noqa: E731
    monkeypatch.setattr(azure_openai, "require", fake)
    monkeypatch.setattr("openbtk.llms.openai.require", fake)
    return module, client


def test_it_is_registered_and_is_an_openai_provider() -> None:
    assert "llm.general.azure_openai" in LLM_REGISTRY.list_keys()
    assert issubclass(AzureOpenAIProvider, OpenAIProvider)
    assert AzureOpenAIProvider.sends_data_offsite is True


def test_the_deployment_name_is_required() -> None:
    with pytest.raises(TypeError):
        AzureOpenAIProvider()  # type: ignore[call-arg]


def test_constructing_does_no_io(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(name: str, extra: str) -> None:
        raise AssertionError("constructing must not import or connect")

    monkeypatch.setattr(azure_openai, "require", explode)
    AzureOpenAIProvider(model="d")


def test_the_azure_client_is_built_lazily_with_endpoint_and_version(sdk: Any) -> None:
    module, _ = sdk
    provider = AzureOpenAIProvider(
        model="my-deployment",
        azure_endpoint="https://r.openai.azure.com",
        api_key="k",  # pragma: allowlist secret
        api_version="2024-06-01",
    )
    module.AzureOpenAI.assert_not_called()
    provider.generate("a")
    provider.generate("b")
    module.AzureOpenAI.assert_called_once_with(
        api_key="k",  # pragma: allowlist secret
        azure_endpoint="https://r.openai.azure.com",
        api_version="2024-06-01",
    )


def test_unset_values_are_left_to_the_sdks_own_environment_lookup(sdk: Any) -> None:
    module, _ = sdk
    AzureOpenAIProvider(model="d").generate("x")
    module.AzureOpenAI.assert_called_once_with(
        api_key=None, azure_endpoint=None, api_version=None
    )


def test_the_deployment_name_is_sent_as_the_model(sdk: Any) -> None:
    _, client = sdk
    result = AzureOpenAIProvider(model="my-deployment").generate("hi")
    assert client.chat.completions.create.call_args.kwargs["model"] == "my-deployment"
    assert result.text == "hello" and result.usage is not None
    assert result.usage.total_tokens == 3


def test_openai_errors_are_translated_as_in_the_parent(sdk: Any) -> None:
    _, client = sdk
    client.chat.completions.create.side_effect = _AuthError("denied")
    with pytest.raises(AuthenticationError):
        AzureOpenAIProvider(model="d", max_retries=1).generate("x")
    client.chat.completions.create.side_effect = _RateError("slow")
    with pytest.raises(RateLimitError):
        AzureOpenAIProvider(model="d", max_retries=1).generate("x")


def test_provenance_records_the_deployment_but_never_the_key() -> None:
    provider = AzureOpenAIProvider(
        model="my-deployment",
        azure_endpoint="https://r.openai.azure.com",
        api_key="SECRET-KEY-VALUE",  # pragma: allowlist secret
        api_version="2024-06-01",
    )
    provenance = provider.provenance()
    assert provenance.config == {
        "deployment": "my-deployment",
        "azure_endpoint": "https://r.openai.azure.com",
        "api_version": "2024-06-01",
    }
    assert "SECRET-KEY-VALUE" not in provenance.model_dump_json()
    identity = provenance.model_identity
    assert identity is not None
    assert (identity.name, identity.source) == ("my-deployment", "azure")
