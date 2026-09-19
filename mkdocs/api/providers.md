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
