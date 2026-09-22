"""Streaming DAG executor, pipeline state, and built-in recipes.

Owned by OpenBTK; requires no external orchestration framework (ADR-0001).

``RAGPipeline`` (task 5.8) is a separate, non-executor orchestration path
for the query side of a RAG system -- see its own module docstring for
why it is not another executor step type.

``join_notes_to_events`` (FR-E-08) joins notes to a patient's structured events; it
lives here for the same cross-modal reason.

``Pipeline.run(checkpoint_path=...)`` (FR-L-05) resumes a long run instead of
restarting it; see ``openbtk.pipelines.checkpoint`` for exactly what that
guarantees.

``PatientTimelineSerializer`` (task 6.5) lives here rather than under
``openbtk.data.ehr`` for the same structural reason: it is a cross-modal
converter that genuinely needs both ``openbtk.data.ehr`` and
``openbtk.data.clinical_text`` concrete types, and import-linter's
"Modalities are independent of one another" contract forbids one modality
package importing another directly. ``openbtk.pipelines`` sits above
``openbtk.data`` in the layered-architecture contract and may depend on
either -- see its own module docstring for the full reasoning.
"""

from __future__ import annotations

from openbtk.pipelines.checkpoint import Checkpoint, load_checkpoint, save_checkpoint
from openbtk.pipelines.join import (
    MatchedEvent,
    NoteWithEvents,
    index_patients,
    join_notes_to_events,
)
from openbtk.pipelines.pipeline import Pipeline, Step
from openbtk.pipelines.rag import RAGPipeline
from openbtk.pipelines.timeline import PatientTimelineSerializer

__all__ = [
    "Checkpoint",
    "MatchedEvent",
    "NoteWithEvents",
    "PatientTimelineSerializer",
    "Pipeline",
    "RAGPipeline",
    "Step",
    "index_patients",
    "join_notes_to_events",
    "load_checkpoint",
    "save_checkpoint",
]
