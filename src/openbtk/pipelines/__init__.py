"""Streaming DAG executor, pipeline state, and built-in recipes.

Owned by OpenBTK; requires no external orchestration framework (ADR-0001).

``RAGPipeline`` (task 5.8) is a separate, non-executor orchestration path
for the query side of a RAG system -- see its own module docstring for
why it is not another executor step type.
"""

from __future__ import annotations

from openbtk.pipelines.pipeline import Pipeline, Step
from openbtk.pipelines.rag import RAGPipeline

__all__ = ["Pipeline", "RAGPipeline", "Step"]
