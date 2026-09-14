# Clinical text

`openbtk.data.clinical_text` must be imported explicitly — it registers its
loaders, preprocessors and chunkers as a side effect. None of them need
their optional dependency (medspaCy, spaCy, pandas) merely to be *defined*,
only to actually run.

## Schemas

::: openbtk.data.clinical_text.schemas.ClinicalTextRecord

::: openbtk.data.clinical_text.schemas.ClinicalTextChunk

## Loaders

::: openbtk.data.clinical_text.loaders.PlainTextLoader

::: openbtk.data.clinical_text.loaders.JSONLLoader

::: openbtk.data.clinical_text.loaders.MIMICNotesLoader

## Preprocessing

::: openbtk.data.clinical_text.preprocessing.SectionSegmenter

::: openbtk.data.clinical_text.preprocessing.DeidPreprocessor

## Chunking

The one place OpenBTK writes genuinely new logic, not a wrapped library —
generic chunkers destroy clinical meaning by splitting mid-section.

::: openbtk.data.clinical_text.chunking.SectionAwareChunker

::: openbtk.data.clinical_text.chunking.FixedTokenChunker

## Tokenization

::: openbtk.data.clinical_text.tokenization.count_tokens_approximate

::: openbtk.data.clinical_text.tokenization.count_tokens_exact
