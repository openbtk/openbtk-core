from pydantic import BaseModel, Field
from typing import Dict, Any, List

class MedChunk(BaseModel):
    """The canonical output object for every processed data chunk."""
    id: str
    patient_id: str
    encounter_id: str
    modality: str
    text_content: str = ""
    raw_data_link: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    vectors: Dict[str, List[float]] = Field(default_factory=dict)
    # Example: vectors = {"text_vec": [...], "audio_vec": [...]}

class IngestionResult(BaseModel):
    """Result of processing a single raw file."""
    chunks: List[MedChunk]
    success: bool
    error: str = ""