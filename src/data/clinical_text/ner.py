"""
Clinical named entity recognition and vocabulary linking.

- ClinicalEntityLinker: wraps scispacy NER + UMLS entity linking
- NegationDetector: wraps medspacy context (negation/uncertainty/family history)

Both operate on ClinicalTextChunk objects, populating the `entities` field
with LinkedEntity records carrying CUI/SNOMED/ICD codes where available.
"""
from __future__ import annotations

from typing import Any

import structlog

from opentbtk.core.errors import ProcessingError
from opentbtk.core.registry import Registry
from opentbtk.core.schemas import LinkedEntity
from .schemas import ClinicalTextChunk

log = structlog.get_logger(__name__)

# Dedicated registry for NER components — text-modality specific.
NER_REGISTRY: Registry["BaseClinicalNER"] = Registry("ner")


class BaseClinicalNER:
    """Interface for clinical NER / entity linking components."""

    def extract(self, chunk: ClinicalTextChunk) -> ClinicalTextChunk:
        """Return a new chunk with `entities` populated."""
        raise NotImplementedError


_SCISPACY_MODEL_PRESETS: dict[str, str] = {
    "sm": "en_core_sci_sm",
    "md": "en_core_sci_md",
    "lg": "en_core_sci_lg",
    "bionlp13cg": "en_ner_bionlp13cg_md",  # cancer genetics NER
    "bc5cdr": "en_ner_bc5cdr_md",          # chemical/disease NER
}


@NER_REGISTRY.register("ner.clinical_text.entity_linker")
class ClinicalEntityLinker(BaseClinicalNER):
    """Extract and link clinical entities using scispacy + UMLS.

    Wraps a scispacy pipeline with the `scispacy_linker` component to map
    detected entity spans to UMLS Concept Unique Identifiers (CUIs), which
    in turn carry cross-references to SNOMED CT and ICD-10 where available
    in the UMLS metathesaurus.

    Requires scispacy model packages to be installed separately, e.g.:
        pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/
            releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

    Args:
        model_preset: One of "sm", "md", "lg", "bionlp13cg", "bc5cdr".
            Ignored if `model_name` is given.
        model_name: Explicit scispacy/spaCy model name. Takes precedence
            over `model_preset`.
        link_to_umls: If True, adds the UMLS entity linker component
            (requires `scispacy[umls]` extra and downloads a large knowledge
            base on first use — several GB). Default False to keep startup
            fast; entities are still extracted without UMLS linking.
        linker_threshold: Minimum similarity score for UMLS candidate
            acceptance (0–1).
    """

    def __init__(
        self,
        model_preset: str = "sm",
        model_name: str | None = None,
        link_to_umls: bool = False,
        linker_threshold: float = 0.85,
    ) -> None:
        if model_name is None:
            if model_preset not in _SCISPACY_MODEL_PRESETS:
                raise ProcessingError(
                    f"Unknown scispacy model preset '{model_preset}'. "
                    f"Available: {sorted(_SCISPACY_MODEL_PRESETS)}",
                )
            model_name = _SCISPACY_MODEL_PRESETS[model_preset]
        self._model_name = model_name
        self._link_to_umls = link_to_umls
        self._linker_threshold = linker_threshold
        self._nlp: Any = None

    def _ensure_loaded(self) -> None:
        if self._nlp is not None:
            return
        try:
            import spacy
        except ImportError as e:
            raise ProcessingError(
                "spacy/scispacy is required for ClinicalEntityLinker. "
                "Install with: pip install opentbtk[clinical_text]",
            ) from e
        try:
            self._nlp = spacy.load(self._model_name)
        except OSError as e:
            raise ProcessingError(
                f"scispacy model '{self._model_name}' is not installed. "
                f"Download it from the scispacy model releases page.",
                context={"model_name": self._model_name},
            ) from e

        if self._link_to_umls:
            try:
                self._nlp.add_pipe(
                    "scispacy_linker",
                    config={
                        "resolve_abbreviations": True,
                        "linker_name": "umls",
                        "threshold": self._linker_threshold,
                    },
                )
                log.info("ner.umls_linker_loaded", model_name=self._model_name)
            except Exception as e:
                raise ProcessingError(
                    "Failed to load UMLS linker. Ensure scispacy[umls] extra "
                    "is installed (downloads ~1GB knowledge base on first use).",
                ) from e

        log.info("ner.loaded", model_name=self._model_name, umls=self._link_to_umls)

    def extract(self, chunk: ClinicalTextChunk) -> ClinicalTextChunk:
        """Extract entities from a chunk and return an updated copy.

        Args:
            chunk: A ClinicalTextChunk with `text` populated.

        Returns:
            A new ClinicalTextChunk with `entities` populated. CUI/SNOMED/
            ICD fields are only populated if `link_to_umls=True` was set
            and the UMLS linker found candidate concepts above threshold.

        Raises:
            ProcessingError: If the scispacy pipeline fails.
        """
        self._ensure_loaded()
        try:
            doc = self._nlp(chunk.text)
        except Exception as e:
            raise ProcessingError(
                "Entity extraction failed",
                context={"chunk_id": chunk.chunk_id},
            ) from e

        entities: list[LinkedEntity] = []
        for ent in doc.ents:
            cui = None
            if self._link_to_umls and hasattr(ent._, "kb_ents") and ent._.kb_ents:
                # Highest-scoring UMLS candidate
                cui, _score = ent._.kb_ents[0]

            entities.append(LinkedEntity(
                text=ent.text,
                start=ent.start_char,
                end=ent.end_char,
                label=ent.label_,
                cui=cui,
            ))

        log.debug(
            "ner.extract.complete",
            chunk_id=chunk.chunk_id,
            n_entities=len(entities),
        )
        return chunk.model_copy(update={"entities": entities})


@NER_REGISTRY.register("ner.clinical_text.negation_detector")
class NegationDetector(BaseClinicalNER):
    """Detect negation, uncertainty, and family-history context for entities.

    Wraps medspacy's `context` component (an implementation of the ConText
    algorithm, an extension of NegEx) to determine whether each previously
    extracted entity is negated (e.g., "no evidence of pneumonia"), uncertain
    ("possible pneumonia"), or pertains to family history ("father has
    diabetes") rather than the patient.

    Must run AFTER ClinicalEntityLinker has populated `chunk.entities`,
    since this component annotates existing entities rather than detecting
    new ones.

    Args:
        rules: Optional path to a custom medspacy context rules file. If
            None, uses medspacy's default English clinical rule set.
    """

    def __init__(self, rules: str | None = None) -> None:
        self._rules_path = rules
        self._nlp: Any = None

    def _ensure_loaded(self) -> None:
        if self._nlp is not None:
            return
        try:
            import medspacy
            from medspacy.context import ConTextComponent
        except ImportError as e:
            raise ProcessingError(
                "medspacy is required for NegationDetector. "
                "Install with: pip install opentbtk[clinical_text]",
            ) from e

        self._nlp = medspacy.load(enable=["sentencizer", "target_matcher", "context"])
        log.info("ner.negation_detector.loaded")

    def extract(self, chunk: ClinicalTextChunk) -> ClinicalTextChunk:
        """Annotate chunk entities with negation/uncertainty/family-history flags.

        Args:
            chunk: A ClinicalTextChunk with `entities` already populated
                (run ClinicalEntityLinker first).

        Returns:
            A new ClinicalTextChunk with each entity's metadata enriched.
            Since LinkedEntity is frozen and has no native negation field,
            results are attached via `chunk.metadata["negation"]` keyed by
            entity span (start, end) -> {"negated": bool, "uncertain": bool,
            "family_history": bool}.

        Raises:
            ProcessingError: If the medspacy pipeline fails.
        """
        self._ensure_loaded()
        if not chunk.entities:
            log.debug("ner.negation.skip_no_entities", chunk_id=chunk.chunk_id)
            return chunk

        try:
            doc = self._nlp(chunk.text)
        except Exception as e:
            raise ProcessingError(
                "Negation detection failed",
                context={"chunk_id": chunk.chunk_id},
            ) from e

        # Build a lookup of medspacy target spans to their context flags
        span_flags: dict[tuple[int, int], dict[str, bool]] = {}
        for target in doc.ents:
            span_flags[(target.start_char, target.end_char)] = {
                "negated": bool(getattr(target._, "is_negated", False)),
                "uncertain": bool(getattr(target._, "is_uncertain", False)),
                "family_history": bool(getattr(target._, "is_family", False)),
            }

        negation_metadata: dict[str, dict[str, bool]] = {}
        for entity in chunk.entities:
            key = (entity.start, entity.end)
            if key in span_flags:
                negation_metadata[f"{entity.start}:{entity.end}"] = span_flags[key]

        log.debug(
            "ner.negation.complete",
            chunk_id=chunk.chunk_id,
            n_flagged=len(negation_metadata),
        )
        return chunk.model_copy(
            update={"metadata": {**chunk.metadata, "negation": negation_metadata}}
        )
