"""NER-based PHI recognizer -- statistical named-entity recognition,
covering the two categories ``RuleRecognizer``'s own module docstring says
it cannot: ``NAME`` and ``GEOGRAPHIC_SUBDIVISION`` (ADR-0006's evidence:
"rules miss novel formats, NER misses structured identifiers" -- this is
the other half of that trade-off).

**Implementation choice, stated plainly.** ADR-0006 names "scispaCy /
transformers" as the intended NER backend. This uses spaCy's own
general-purpose small English pipeline (``en_core_web_sm``) directly
instead: scispaCy's off-the-shelf biomedical models are trained to find
scientific entities (DISEASE, CHEMICAL, GENE, ...), not PERSON or GPE --
exactly the two categories this recognizer exists to cover -- so they
would not actually help here, benchmarked claims aside. ``en_core_web_sm``
is also two orders of magnitude lighter than a transformer NER pipeline
(~12MB, no ``torch``, no GPU) while doing the specific job asked of it.
Revisit this choice if a future clinical corpus exhibits entity shapes
this general-purpose model handles badly -- verified honestly, not
assumed: see ``tests/accuracy/`` for what it actually achieves.

**Confidence is a fixed, documented estimate, not a calibrated one.**
spaCy's small pipeline exposes no native per-entity confidence score
without pipeline surgery this milestone does not attempt. CLAUDE.md rule
14 forbids treating a made-up number as a benchmarked one, so
``_CONFIDENCE`` is labelled for exactly what it is.

**Lazy, per CLAUDE.md rules 5 and 11.** ``spacy`` is an optional
dependency (``pip install openbtk[text]``); importing this module needs
none of it (the ``require()`` call happens inside ``_get_model()``, not at
module scope). The model itself loads once, on the first ``detect()``
call, cached at module level -- never in ``__init__``, so constructing a
``NERRecognizer`` does no I/O and costs nothing whether or not spaCy is
installed.

**Never imported by ``openbtk.deid.recognizers.__init__``** (unlike
``RuleRecognizer``, which is eager because it is dependency-free): the
whole test suite would otherwise need spaCy and a downloaded model just to
import ``openbtk.deid``, breaking the zero-extras install/test story
(docs/07_TEST_CHARTER.md section 3.8). ``DeidEngine`` imports this module
lazily, only when ``"ner"`` is actually requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from openbtk.core._lazy import require
from openbtk.core.errors import DeidError
from openbtk.core.schemas import TextSpan
from openbtk.deid.recognizers.base import RECOGNIZER_REGISTRY, BaseRecognizer
from openbtk.deid.schemas import Detection, PHICategory

if TYPE_CHECKING:
    from spacy.language import Language

_MODEL_NAME = "en_core_web_sm"

_CONFIDENCE = 0.75
"""A documented default, not a calibrated value -- see module docstring."""

_LABEL_MAP: dict[str, PHICategory] = {
    "PERSON": PHICategory.NAME,
    "GPE": PHICategory.GEOGRAPHIC_SUBDIVISION,
    "LOC": PHICategory.GEOGRAPHIC_SUBDIVISION,
    "FAC": PHICategory.GEOGRAPHIC_SUBDIVISION,
}
"""spaCy entity labels this recognizer acts on. Every other label
(ORG, DATE, CARDINAL, ...) is intentionally ignored -- RuleRecognizer
already covers DATE with a real format, and mapping ORG broadly would
trade precision for a category (organisation name) that isn't itself one
of the 18 Safe Harbor identifiers."""

_model_cache: Language | None = None


def _get_model() -> Language:
    """Lazily load and cache the spaCy pipeline. Not called from
    ``__init__`` -- CLAUDE.md rule 11: constructors do no I/O."""
    global _model_cache
    if _model_cache is None:
        spacy = require("spacy", extra="text")
        try:
            _model_cache = spacy.load(_MODEL_NAME)
        except OSError as e:
            raise DeidError(
                f"spaCy model {_MODEL_NAME!r} is required but not "
                f"downloaded. Install it with: "
                f"python -m spacy download {_MODEL_NAME}",
                context={"model": _MODEL_NAME},
            ) from e
    return _model_cache


@RECOGNIZER_REGISTRY.register("recognizer.general.ner")
class NERRecognizer(BaseRecognizer):
    """spaCy-based NER for ``NAME`` and ``GEOGRAPHIC_SUBDIVISION``. See
    module docstring for the model choice and its limitations."""

    method: ClassVar[Literal["rule", "ner", "llm_verifier"]] = "ner"
    requires_model_download: ClassVar[bool] = True

    def detect(self, text: str) -> list[Detection]:
        model = _get_model()
        doc = model(text)
        detections: list[Detection] = []
        for ent in doc.ents:
            category = _LABEL_MAP.get(ent.label_)
            if category is None:
                continue
            detections.append(
                Detection(
                    category=category,
                    span=TextSpan(
                        start=ent.start_char,
                        end=ent.end_char,
                        label=category.value,
                        confidence=_CONFIDENCE,
                    ),
                    confidence=_CONFIDENCE,
                    method=self.method,
                )
            )
        return detections
