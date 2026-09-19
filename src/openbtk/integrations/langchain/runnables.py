"""OpenBTK components as LangChain ``Runnable`` s, and back.

``as_runnable`` turns a component into a ``Runnable`` with the input and output
shown in the table below; ``from_runnable`` turns a text-to-text ``Runnable``
into a record preprocessor.

===========================  ===========================  ========================
Component                    Input                        Output
===========================  ===========================  ========================
``BaseLLMProvider``          ``str``                      ``str`` (the completion)
``BaseEmbeddingProvider``    ``list[str]``                ``list[list[float]]``
``BasePreprocessor``         a record                     the processed record
``BaseChunker``              a record                     ``list`` of chunks
``BaseGuardrail``            the payload                  ``GuardrailResult``
``DeidEngine``               ``{"text", "patient_id"}``   ``DeidResult``
===========================  ===========================  ========================

A chunker's output is a list because a ``Runnable`` returns one value; one
record's chunks are materialised together, which is the same bound the chunker
already has per record.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import Runnable, RunnableLambda

from openbtk.core.base import (
    BaseChunker,
    BaseEmbeddingProvider,
    BaseGuardrail,
    BaseLLMProvider,
    BasePreprocessor,
    Component,
)
from openbtk.core.errors import ConfigError, ProcessingError
from openbtk.data.clinical_text.schemas import ClinicalTextRecord
from openbtk.deid.engine import DeidEngine
from openbtk.integrations.langchain.embeddings import OpenBTKEmbeddings

_SUPPORTED = (
    "BaseLLMProvider, BaseEmbeddingProvider, BasePreprocessor, BaseChunker, "
    "BaseGuardrail, DeidEngine"
)


def as_runnable(component: Component | DeidEngine) -> Runnable[Any, Any]:
    """Wrap ``component`` as a LangChain ``Runnable`` (see the module table).

    Raises:
        ConfigError: If ``component`` is not one of the supported kinds.

    Example:
        >>> import contextlib, io
        >>> with contextlib.redirect_stdout(io.StringIO()):  # registry debug logs
        ...     from openbtk.guardrails.phi_leakage import PHILeakageGuardrail
        ...     result = as_runnable(PHILeakageGuardrail()).invoke("Plan: rest.")
        >>> result.passed
        True
    """
    name = type(component).__name__
    fn: Any
    if isinstance(component, BaseLLMProvider):
        llm = component

        def fn(prompt: str) -> str:
            return llm.generate(prompt).text

    elif isinstance(component, BaseEmbeddingProvider):
        embeddings = OpenBTKEmbeddings(component)
        fn = embeddings.embed_documents
    elif isinstance(component, BasePreprocessor):
        fn = component.process
    elif isinstance(component, BaseChunker):
        chunker = component

        def fn(record: Any) -> list[Any]:
            return list(chunker.chunk(record))

    elif isinstance(component, BaseGuardrail):
        fn = component.check
    elif isinstance(component, DeidEngine):
        engine = component

        def fn(payload: dict[str, str]) -> Any:
            try:
                text, patient_id = payload["text"], payload["patient_id"]
            except (KeyError, TypeError) as e:
                raise ConfigError(
                    'A DeidEngine runnable takes {"text": ..., "patient_id": ...}.'
                ) from e
            return engine.deidentify(text, patient_id=patient_id)

    else:
        raise ConfigError(
            f"as_runnable() does not support {name}. Supported: {_SUPPORTED}.",
            context={"component": name},
        )
    return RunnableLambda(fn, name=name)


class RunnablePreprocessor(BasePreprocessor[ClinicalTextRecord]):
    """A text-to-text ``Runnable`` applied to each record's ``text``.

    Not registered under a registry key: it wraps a live object, which a
    config file cannot express. Use it programmatically, e.g. via
    ``process_stream``. A rewrite that changes the text drops ``sections``
    (their offsets no longer describe the new text); ``deid_status`` is left as
    it was -- rewriting text does not make a record de-identified.

    Example:
        >>> from langchain_core.runnables import RunnableLambda
        >>> rec = ClinicalTextRecord(record_id="r1", source="s", text="abc")
        >>> pre = from_runnable(RunnableLambda(str.upper))
        >>> pre.process(rec).text
        'ABC'
    """

    def __init__(self, runnable: Runnable[str, str]) -> None:
        self._runnable = runnable

    def process(self, record: ClinicalTextRecord) -> ClinicalTextRecord:
        result = self._runnable.invoke(record.text)
        if not isinstance(result, str):
            raise ProcessingError(
                f"The wrapped Runnable must return str, got {type(result).__name__}.",
                context={"record_id": record.record_id},
            )
        if result == record.text:
            return record
        return record.model_copy(update={"text": result, "sections": None})


def from_runnable(runnable: Runnable[str, str]) -> RunnablePreprocessor:
    """Wrap a text-to-text ``Runnable`` as an OpenBTK record preprocessor."""
    return RunnablePreprocessor(runnable)
