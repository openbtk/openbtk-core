"""Azure OpenAI LLM provider (FR-V-01): the OpenAI provider on an Azure resource.

Azure serves OpenAI's models from *your own* resource, under a *deployment name* you
chose, so the only differences from :class:`~openbtk.llms.openai.OpenAIProvider` are how
the SDK client is built and what ``model`` means: here it is the **deployment name**,
not an OpenAI model id. Request, response, streaming, retry and error translation are
all inherited.

Credentials are the SDK's to resolve and are never read, held or logged here: pass
``api_key=`` or leave it ``None`` so the SDK reads ``AZURE_OPENAI_API_KEY``. The
endpoint and API version resolve the same way (``AZURE_OPENAI_ENDPOINT`` and
``OPENAI_API_VERSION``). Microsoft Entra ID token auth is not wrapped yet.

This provider **sends data off the machine** (to your Azure resource), like every hosted
provider, so it is refused unless the policy allows off-site providers. It is tested
against a mocked SDK; it has not been run against a live Azure resource.
"""

from __future__ import annotations

from typing import Any

from openbtk.core._lazy import require
from openbtk.core.provenance import ComponentProvenance, ModelIdentity
from openbtk.core.registry import LLM_REGISTRY
from openbtk.llms.openai import OpenAIProvider


@LLM_REGISTRY.register("llm.general.azure_openai")
class AzureOpenAIProvider(OpenAIProvider):
    """Generate text through an Azure OpenAI deployment.

    Args:
        model: The **deployment name** (what you named the deployment in Azure), which
               the API takes where OpenAI takes a model id.
        azure_endpoint: ``https://<resource>.openai.azure.com``. ``None`` lets the SDK
                        read ``AZURE_OPENAI_ENDPOINT``.
        api_key: Forwarded to the SDK. ``None`` lets it read ``AZURE_OPENAI_API_KEY``.
        api_version: The Azure API version. ``None`` lets the SDK read
            ``OPENAI_API_VERSION``.
        max_retries: Rate-limit retry attempts (see the OpenAI provider).

    Example:
        >>> provider = AzureOpenAIProvider(model="my-deployment")  # nothing is called
        >>> provider.sends_data_offsite
        True
    """

    def __init__(
        self,
        *,
        model: str,
        azure_endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        max_retries: int = 5,
    ) -> None:
        super().__init__(model=model, api_key=api_key, max_retries=max_retries)
        self._azure_endpoint = azure_endpoint
        self._api_version = api_version

    def _get_client(self) -> Any:
        if self._client is None:
            openai = require("openai", extra="llms")
            self._client = openai.AzureOpenAI(
                api_key=self._api_key,
                azure_endpoint=self._azure_endpoint,
                api_version=self._api_version,
            )
        return self._client

    def model_identity(self) -> ModelIdentity:
        return ModelIdentity(name=self._model, revision=self._model, source="azure")

    def provenance(self) -> ComponentProvenance:
        return (
            super()
            .provenance()
            .model_copy(
                update={
                    "config": {
                        "deployment": self._model,
                        "azure_endpoint": self._azure_endpoint,
                        "api_version": self._api_version,
                    },
                    "model_identity": self.model_identity(),
                }
            )
        )
