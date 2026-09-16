"""LLM providers. Biomedical models are configured presets of the HuggingFace
and endpoint providers, not bespoke classes.

``llms.base`` holds the shared building blocks (message/response/token-usage
re-exports plus :func:`~openbtk.llms.base.retry_with_backoff`) every
concrete provider is built from. No submodule is imported here for its
registration side effect yet -- there is nothing to register until task 5.2
adds a concrete provider, unlike ``data.clinical_text``'s ``__init__.py``.
"""
