"""The cloud providers against the *real* SDK libraries (no network or credentials).

The provider unit tests use fake modules, which prove the adapters' logic but not that
the shapes they build are ones the real libraries accept. These tests close that gap
offline: Bedrock's request is checked by botocore's own parameter validator, the Vertex
request is built from real ``google.genai`` types, real SDK error objects are
translated, and the Azure client is constructed for real. Each skips when its library is
not installed; the CI job ``test-cloud-sdks`` installs them all and fails if any test is
skipped.

They cannot show that a live service *accepts* a request. None of these providers has
been run against a live AWS, Google Cloud or Azure account.
"""

from __future__ import annotations

from typing import Any

import pytest

from openbtk.core.errors import AuthenticationError, ProviderError, RateLimitError
from openbtk.core.schemas import Message

_MESSAGES = [
    Message(role="system", content="Be brief."),
    Message(role="user", content="What is 2+2?"),
    Message(role="assistant", content="4"),
    Message(role="user", content="And 3+3?"),
]


class TestBedrockAgainstBotocore:
    def _service(self) -> Any:
        botocore_session = pytest.importorskip("botocore.session")
        return botocore_session.get_session().get_service_model("bedrock-runtime")

    def test_the_converse_request_passes_botocores_own_validator(self) -> None:
        shape = self._service().operation_model("Converse").input_shape
        from botocore.validate import validate_parameters

        from openbtk.llms.bedrock import BedrockProvider

        provider = BedrockProvider(model="vendor.model-v1:0")
        request = provider._request(
            _MESSAGES,
            {"max_tokens": 64, "temperature": 0.2, "top_p": 0.9, "stop": ["END"]},
        )
        validate_parameters(request, shape)

    def test_the_minimal_request_passes_the_validator_too(self) -> None:
        shape = self._service().operation_model("Converse").input_shape
        from botocore.validate import validate_parameters

        from openbtk.llms.bedrock import BedrockProvider

        request = BedrockProvider(model="m")._request(
            [Message(role="user", content="hi")], {}
        )
        validate_parameters(request, shape)

    def test_a_setting_the_service_does_not_define_would_be_caught(self) -> None:
        """Proves the validator is really checking: an unknown field is rejected."""
        shape = self._service().operation_model("Converse").input_shape
        from botocore.exceptions import ParamValidationError
        from botocore.validate import validate_parameters

        with pytest.raises(ParamValidationError):
            validate_parameters({"modelId": "m", "messages": [], "nope": 1}, shape)

    def test_the_inference_config_fields_exist_in_the_service_model(self) -> None:
        from openbtk.llms.bedrock import _INFERENCE_KEYS

        members = self._service().operation_model("Converse").input_shape.members
        assert set(_INFERENCE_KEYS.values()) <= set(members["inferenceConfig"].members)

    def test_real_client_errors_are_translated(self) -> None:
        exceptions = pytest.importorskip("botocore.exceptions")
        from openbtk.llms.bedrock import BedrockProvider

        def client_error(code: str) -> Any:
            return exceptions.ClientError(
                {"Error": {"Code": code, "Message": "m"}}, "Converse"
            )

        translate = BedrockProvider._translate
        assert isinstance(
            translate(exceptions, client_error("ThrottlingException")), RateLimitError
        )
        assert isinstance(
            translate(exceptions, client_error("AccessDeniedException")),
            AuthenticationError,
        )
        assert isinstance(
            translate(exceptions, client_error("ValidationException")), ProviderError
        )
        assert isinstance(
            translate(exceptions, exceptions.NoCredentialsError()), AuthenticationError
        )
        assert translate(exceptions, KeyError("x")) is None

    def test_the_throttling_and_auth_codes_are_real_bedrock_error_names(self) -> None:
        from openbtk.llms.bedrock import _AUTH, _THROTTLING

        declared = {
            e.name for e in self._service().operation_model("Converse").error_shapes
        }
        # Codes the Converse operation declares; the rest (expired token and so on) are
        # generic AWS error codes returned by the platform, not modelled per operation.
        assert "ThrottlingException" in declared and "AccessDeniedException" in declared
        assert {"ThrottlingException"} <= _THROTTLING and {
            "AccessDeniedException"
        } <= _AUTH


class TestVertexAgainstGoogleGenai:
    def test_the_request_is_built_from_real_sdk_types(self) -> None:
        pytest.importorskip("google.genai")
        from google.genai import types

        from openbtk.llms.vertex import VertexAIProvider

        request = VertexAIProvider(model="model-001")._request(
            _MESSAGES,
            {"max_tokens": 64, "temperature": 0.2, "top_p": 0.9, "stop": ["END"]},
        )
        assert all(isinstance(c, types.Content) for c in request["contents"])
        assert [c.role for c in request["contents"]] == ["user", "model", "user"]
        config = request["config"]
        assert isinstance(config, types.GenerateContentConfig)
        assert config.system_instruction == "Be brief."
        assert (config.max_output_tokens, config.stop_sequences) == (64, ["END"])

    def test_the_sdk_rejects_an_unknown_setting_so_ours_are_real_fields(self) -> None:
        pytest.importorskip("google.genai")
        from google.genai import types

        for name in (
            "max_output_tokens",
            "temperature",
            "top_p",
            "stop_sequences",
            "system_instruction",
        ):
            assert name in types.GenerateContentConfig.model_fields

    def test_the_usage_fields_read_exist_on_the_real_response_type(self) -> None:
        pytest.importorskip("google.genai")
        from google.genai import types

        fields = types.GenerateContentResponseUsageMetadata.model_fields
        assert {
            "prompt_token_count",
            "candidates_token_count",
            "total_token_count",
        } <= set(fields)
        assert hasattr(types.GenerateContentResponse, "text")

    def test_the_client_takes_the_vertex_arguments(self) -> None:
        import inspect

        genai = pytest.importorskip("google.genai")
        parameters = inspect.signature(genai.Client.__init__).parameters
        assert {"vertexai", "project", "location"} <= set(parameters)

    def test_real_api_errors_are_translated(self) -> None:
        pytest.importorskip("google.genai")
        from google.genai import errors

        from openbtk.llms.vertex import VertexAIProvider

        def api_error(code: int) -> Any:
            return errors.APIError(
                code, {"error": {"code": code, "message": "m", "status": "S"}}
            )

        translate = VertexAIProvider._translate
        assert isinstance(translate(errors, api_error(429)), RateLimitError)
        assert isinstance(translate(errors, api_error(403)), AuthenticationError)
        assert isinstance(translate(errors, api_error(500)), ProviderError)
        assert translate(errors, KeyError("x")) is None


class TestAzureAgainstOpenAI:
    def test_the_real_azure_client_accepts_our_arguments(self) -> None:
        openai = pytest.importorskip("openai")
        client = openai.AzureOpenAI(
            api_key="not-a-real-key",  # pragma: allowlist secret
            azure_endpoint="https://resource.openai.azure.com",
            api_version="2024-06-01",
        )
        assert "resource.openai.azure.com" in str(client.base_url)

    def test_the_provider_builds_the_real_client_lazily(self) -> None:
        pytest.importorskip("openai")
        from openbtk.llms.azure_openai import AzureOpenAIProvider

        provider = AzureOpenAIProvider(
            model="my-deployment",
            azure_endpoint="https://resource.openai.azure.com",
            api_key="not-a-real-key",  # pragma: allowlist secret
            api_version="2024-06-01",
        )
        assert type(provider._get_client()).__name__ == "AzureOpenAI"
