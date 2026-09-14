"""``BaseRecognizer``: the pluggable extension point for PHI detection
(ADR-0006).

A recognizer's whole job is "find spans in this text, tag them with a
category and a confidence". Combining several recognizers' results into one
final answer is ``SpanMerger``'s job (``openbtk.deid.merger``), not this
class's -- keeping recognizers single-purpose is what makes adding an
institution-specific one (a novel MRN format, say) "register a class", not
"patch the engine" (ADR-0006's stated design goal).

Lives in ``openbtk.deid``, not ``openbtk.core``: recognizers are a
de-identification-specific extension point, not a general one every OpenBTK
component needs. ``BaseRecognizer`` still subclasses ``Component``
(``openbtk.core.base``) to get the same identity/provenance contract every
other extension point has -- that dependency runs downward (deid depends on
core), which the layering rule allows.

``RECOGNIZER_REGISTRY`` lives here too, as its own ``Registry`` instance --
deliberately NOT one of core's 12 global registries or reachable through
``get_registry()``: core must never import from ``openbtk.deid``
(docs/03_ARCHITECTURE.md section 2). ``"recognizer"`` is reserved in
``openbtk.core.registry``'s key grammar for exactly this purpose, the same
way ``"finetuner"`` is reserved without a backing registry.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar, Literal

from openbtk.core.base import Component
from openbtk.core.registry import Registry

if TYPE_CHECKING:
    from openbtk.deid.schemas import Detection


class BaseRecognizer(Component):
    """Detect PHI spans in a single document's text.

    Unlike ``BaseLoader``/``BaseChunker``, ``detect()`` returns a plain
    ``list``, not an ``Iterator``: a single document produces a small,
    bounded number of PHI spans (tens, not millions), so there is no
    O(corpus) memory hazard here for ADR-0004's streaming rule to guard
    against -- the corpus-level streaming boundary is the loader's job, one
    document at a time.
    """

    method: ClassVar[Literal["rule", "ner", "llm_verifier"]]
    """Stamped onto every ``Detection`` this recognizer produces
    (docs/04_API_DESIGN.md section 5). ``"ensemble"`` is reserved for
    ``SpanMerger``'s own output and is never a single recognizer's value."""

    @abstractmethod
    def detect(self, text: str) -> list[Detection]:
        """Find every PHI span this recognizer can find in ``text``.

        Args:
            text: The document text to scan. Never logged, never placed in
                a raised exception's message or context.

        Returns:
            Zero or more ``Detection`` objects, in no particular order.
            Confidence is this recognizer's own estimate; final confidence
            combination is ``SpanMerger``'s job, not this method's.

        Raises:
            DeidError: On a recognizer-internal failure. Never returns a
                silently-partial result -- a caller needs to know detection
                was incomplete, not trust an accidentally-short list.
        """


RECOGNIZER_REGISTRY: Registry[BaseRecognizer] = Registry(
    "recognizer",
    BaseRecognizer,  # type: ignore[type-abstract]
)
