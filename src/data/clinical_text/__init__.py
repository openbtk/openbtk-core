"""
OpenTBTK clinical text modality.

Importing this module registers all clinical text components in their
respective global registries. Components are available after import via:
    from opentbtk.core.registry import LOADER_REGISTRY
    loader = LOADER_REGISTRY.create("loader.clinical_text.mimic_notes", ...)

Heavy-dependency components (tokenization, NER, embeddings, fine-tuning)
lazy-load their underlying libraries (transformers, scispacy, medspacy,
peft) on first use, not on import — so importing this module stays fast
even without those extras installed. Calling their methods without the
extras installed raises a clear ProcessingError/EmbeddingError with
install instructions.
"""
from opentbtk.data.clinical_text.schemas import ClinicalTextRecord, ClinicalTextChunk
from opentbtk.data.clinical_text.loaders import (
    PlainTextLoader,
    MIMICNotesLoader,
    CDAClinicalLoader,
)
from opentbtk.data.clinical_text.preprocessing import (
    DeidentifyProcessor,
    SectionSegmenter,
    AbbreviationExpander,
)
from opentbtk.data.clinical_text.chunking import (
    SectionAwareChunker,
    FixedTokenChunker,
    SemanticChunker,
)
from opentbtk.data.clinical_text.guardrails import (
    PHIGuardrail,
    HallucinationGuardrail,
    CodeValidityGuardrail,
)
from opentbtk.data.clinical_text.tokenization import (
    TOKENIZER_REGISTRY,
    BaseTokenizer,
    BiomedicalTokenizer,
)
from opentbtk.data.clinical_text.ner import (
    NER_REGISTRY,
    BaseClinicalNER,
    ClinicalEntityLinker,
    NegationDetector,
)
from opentbtk.data.clinical_text.embeddings import (
    PubMedBERTEmbedding,
    BioBERTEmbedding,
    ClinicalBERTEmbedding,
    SapBERTEmbedding,
)
from opentbtk.data.clinical_text.datasets import (
    MIMICNotesDataset,
    PubMedDataset,
    SyntheaNotesDataset,
)
from opentbtk.data.clinical_text.finetuning import (
    ClinicalFineTuningConfig,
    ClinicalTextFineTuner,
)

__all__ = [
    # Schemas
    "ClinicalTextRecord", "ClinicalTextChunk",
    # Loaders
    "PlainTextLoader", "MIMICNotesLoader", "CDAClinicalLoader",
    # Preprocessing
    "DeidentifyProcessor", "SectionSegmenter", "AbbreviationExpander",
    # Chunking
    "SectionAwareChunker", "FixedTokenChunker", "SemanticChunker",
    # Guardrails
    "PHIGuardrail", "HallucinationGuardrail", "CodeValidityGuardrail",
    # Tokenization
    "TOKENIZER_REGISTRY", "BaseTokenizer", "BiomedicalTokenizer",
    # NER
    "NER_REGISTRY", "BaseClinicalNER", "ClinicalEntityLinker", "NegationDetector",
    # Embeddings
    "PubMedBERTEmbedding", "BioBERTEmbedding", "ClinicalBERTEmbedding", "SapBERTEmbedding",
    # Datasets
    "MIMICNotesDataset", "PubMedDataset", "SyntheaNotesDataset",
    # Fine-tuning
    "ClinicalFineTuningConfig", "ClinicalTextFineTuner",
]
