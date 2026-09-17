"""A local HuggingFace ``transformers`` text-generation LLM provider.

The only provider of the four in task 5.2 that runs entirely on-device --
``sends_data_offsite`` is ``False`` here and ``True`` on the other three.
No retry/backoff applies: there is no remote rate limit to back off from,
so unlike :mod:`openbtk.llms.openai` and :mod:`openbtk.llms.anthropic`,
generation failures are wrapped in :class:`~openbtk.core.errors.ProviderError`
directly rather than routed through :func:`~openbtk.llms.base.retry_with_backoff`.

``revision`` is a *required* constructor argument, not optional with a
silent default -- FR-P-05 (docs/adr, enforced by
:class:`~openbtk.core.provenance.ModelIdentity`) requires an immutable
pinned revision, and unlike an API-based provider (where the model name
itself is the vendor's own pinned unit), a bare HuggingFace model name
with no revision resolves to whatever is on the hub's default branch at
load time -- a real floating target this project should not paper over
with an implicit default.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, ClassVar

from openbtk.core._lazy import require
from openbtk.core.base import BaseLLMProvider
from openbtk.core.errors import ProviderError
from openbtk.core.logging import get_logger
from openbtk.core.provenance import ComponentProvenance, ModelIdentity
from openbtk.core.registry import LLM_REGISTRY
from openbtk.core.schemas import LLMResponse, Message, TokenUsage

if TYPE_CHECKING:
    from collections.abc import Iterator

log = get_logger(__name__)

_DEFAULT_MAX_NEW_TOKENS = 256


@LLM_REGISTRY.register("llm.general.huggingface_local")
class HuggingFaceLocalProvider(BaseLLMProvider):
    """Generate text from a local ``transformers`` causal LM.

    Args:
        model: A HuggingFace Hub model id or local path.
        revision: A pinned commit SHA (never a floating branch name --
            see this module's own docstring for why this has no default).
        device: Passed to the loaded model's ``.to(...)``, e.g. ``"cpu"``,
            ``"cuda"``, ``"mps"``.
        max_new_tokens: Default generation length cap, overridable per call.
        do_sample: Default sampling mode, overridable per call. ``False``
            (greedy decoding) so results are reproducible by default.

    No I/O happens in ``__init__`` (docs/09_CODING_STANDARDS.md rule 11):
    the tokenizer and model are downloaded/loaded lazily, on first real
    call -- genuinely necessary here, not just a style rule, since a model
    download can be gigabytes.
    """

    sends_data_offsite: ClassVar[bool] = False

    def __init__(
        self,
        *,
        model: str,
        revision: str,
        device: str = "cpu",
        max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = False,
    ) -> None:
        self._model_name = model
        self._revision = revision
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._do_sample = do_sample
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self) -> tuple[Any, Any]:
        if self._tokenizer is None or self._model is None:
            require("torch", extra="llms")  # presence check: transformers needs a
            # tensor backend to actually run a model, even though this class
            # never calls torch's API directly (transformers does, internally).
            transformers = require("transformers", extra="text")
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                self._model_name, revision=self._revision
            )
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self._model_name, revision=self._revision
            )
            self._model = model.to(self._device)
        return self._tokenizer, self._model

    def _prompt_from_messages(self, messages: list[Message]) -> str:
        """Render a message list to a single prompt string.

        Uses the tokenizer's own chat template when the model ships one --
        the correct, model-specific way to format a conversation. Falls
        back to a plain role-prefixed transcript when it doesn't (a real,
        disclosed limitation, not a silent guess at the model's expected
        format): many small/base local models have no chat template at
        all, and failing outright would make chat() unusable for them.
        """
        tokenizer, _ = self._load()
        if getattr(tokenizer, "chat_template", None):
            rendered: str = tokenizer.apply_chat_template(
                [{"role": m.role, "content": m.content} for m in messages],
                tokenize=False,
                add_generation_prompt=True,
            )
            return rendered
        return "\n".join(f"{m.role}: {m.content}" for m in messages) + "\nassistant:"

    def _generate_ids(self, prompt: str, **kwargs: Any) -> tuple[Any, Any, int]:
        """Tokenize ``prompt``, run generation, and return
        ``(tokenizer, output_ids, prompt_length)``."""
        tokenizer, model = self._load()
        torch = require("torch", extra="llms")
        inputs = tokenizer(prompt, return_tensors="pt").to(self._device)
        max_new_tokens = kwargs.pop("max_new_tokens", self._max_new_tokens)
        do_sample = kwargs.pop("do_sample", self._do_sample)
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    **kwargs,
                )
        except Exception as e:
            raise ProviderError(f"Local generation failed: {e}") from e
        prompt_length = inputs["input_ids"].shape[-1]
        return tokenizer, output_ids, prompt_length

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        tokenizer, output_ids, prompt_length = self._generate_ids(prompt, **kwargs)
        generated = output_ids[0][prompt_length:]
        text: str = tokenizer.decode(generated, skip_special_tokens=True)
        completion_length = generated.shape[-1]
        usage = TokenUsage(
            prompt_tokens=prompt_length,
            completion_tokens=completion_length,
            total_tokens=prompt_length + completion_length,
        )
        return LLMResponse(text=text, usage=usage)

    def chat(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        return self.generate(self._prompt_from_messages(messages), **kwargs)

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream generated text via a background-thread ``TextIteratorStreamer``
        -- the standard ``transformers`` pattern: ``model.generate()`` is
        synchronous and blocking, so it runs on its own thread while this
        generator yields tokens off the streamer as they arrive.

        ``streamer.end()`` is called unconditionally in the background
        thread's ``finally``, not only on success: ``TextIteratorStreamer``
        only signals completion (unblocking this generator's ``yield from``)
        when ``generate()`` finishes normally through its own hook. If
        ``generate()`` raises before producing anything, nothing would ever
        end the streamer's queue and the consumer would block forever --
        a real hang, not a hypothetical one.
        """
        tokenizer, model = self._load()
        transformers = require("transformers", extra="text")
        inputs = tokenizer(prompt, return_tensors="pt").to(self._device)
        streamer = transformers.TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        max_new_tokens = kwargs.pop("max_new_tokens", self._max_new_tokens)
        do_sample = kwargs.pop("do_sample", self._do_sample)
        generation_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "streamer": streamer,
            **kwargs,
        }
        error: list[Exception] = []

        def _run() -> None:
            try:
                model.generate(**generation_kwargs)
            except Exception as e:  # forwarded to the caller's thread below
                error.append(e)
            finally:
                streamer.end()

        thread = threading.Thread(target=_run)
        thread.start()
        try:
            yield from streamer
        finally:
            thread.join()
        if error:
            raise ProviderError(f"Local generation failed: {error[0]}") from error[0]

    def model_identity(self) -> ModelIdentity:
        return ModelIdentity(
            name=self._model_name, revision=self._revision, source="huggingface"
        )

    def provenance(self) -> ComponentProvenance:
        return (
            super()
            .provenance()
            .model_copy(
                update={
                    "config": {"model": self._model_name, "device": self._device},
                    "model_identity": self.model_identity(),
                }
            )
        )
