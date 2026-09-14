"""Streaming DAG executor, pipeline state, and built-in recipes.

Owned by OpenBTK; requires no external orchestration framework (ADR-0001).
"""

from __future__ import annotations

from openbtk.pipelines.pipeline import Pipeline, Step

__all__ = ["Pipeline", "Step"]
