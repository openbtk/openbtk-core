"""Clinical text: loaders, section segmentation, chunking, tokenization,
entity linking. See docs/05_DATA_MODALITY_SPEC.md section 1.

``loaders`` is imported here (registering all three P0 loaders as a side
effect) rather than left for a caller to import explicitly: none of them
need their optional dependency (``pandas``, for ``MIMICNotesLoader``)
merely to be *defined* -- only ``require()`` inside ``.load()`` does,
resolved lazily on first real use (CLAUDE.md rule 5). This differs from
``openbtk.deid.recognizers``, where ``NERRecognizer`` is deliberately NOT
imported here: that module's own optional dependency (spaCy) is genuinely
needed the moment the CONTRACT SUITE calls a generic `.detect()` on
whatever sample text it has, which would force spaCy on every test run --
loaders don't have that problem, since importing this package costs
nothing regardless of which loaders a caller ends up using.
"""

from __future__ import annotations

from openbtk.data.clinical_text import chunking, loaders, preprocessing

__all__ = ["chunking", "loaders", "preprocessing"]
