# LLM and embedding providers

Every provider declares `sends_data_offsite`; a pipeline policy that forbids
off-site providers refuses to construct one that sets it. In CI the cloud SDKs
and models are mocked; calls against live services run only when you supply keys
or models.

## LLM providers

::: openbtk.llms.openai.OpenAIProvider

::: openbtk.llms.anthropic.AnthropicProvider

::: openbtk.llms.huggingface.HuggingFaceLocalProvider

::: openbtk.llms.openai_compatible.OpenAICompatibleProvider

### Cloud providers

Azure OpenAI, AWS Bedrock and Google Vertex AI. Each is tested against a mocked SDK
*and* against the real SDK libraries offline (request shapes validated by the
libraries themselves); **none has been run against a live account**. They need the
`llms` (Azure), `bedrock` or `vertex` extra, and read credentials from the SDK's own
environment lookup, never from OpenBTK.

::: openbtk.llms.azure_openai.AzureOpenAIProvider

::: openbtk.llms.bedrock.BedrockProvider

::: openbtk.llms.vertex.VertexAIProvider

### Biomedical presets

Presets are configuration over the classes above, not new classes.

::: openbtk.llms.presets.list_llm_presets

::: openbtk.llms.presets.create_llm_preset

## Embedding providers

::: openbtk.embeddings.huggingface.HuggingFaceEmbeddingProvider

::: openbtk.embeddings.openai.OpenAIEmbeddingProvider

### Biomedical presets

::: openbtk.embeddings.presets.list_embedding_presets

::: openbtk.embeddings.presets.create_embedding_preset
