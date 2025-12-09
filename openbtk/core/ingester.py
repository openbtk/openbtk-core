from typing import List, Dict, Callable, Any
from ..models import MedChunk, IngestionResult
from ..processors.audio_processor import AudioProcessor
from ..processors.text_processor import TextProcessor
from ..processors.image_processor import ImageProcessor
from ..encoders.text_encoder import ClinicalTextEncoder
from ..encoders.multimodal_encoder import MultimodalEncoder

# Initialize Processors and Encoders 
PROCESSORS = {
    "audio": AudioProcessor(),
    "text": TextProcessor(),
    "image": ImageProcessor(),
}
ENCODERS = {
    "text_vec": ClinicalTextEncoder(),
    "multimodal": MultimodalEncoder(),
}

class OpenBTKIngester:
    """
    The main API for OpenBTK. Orchestrates chunking and encoding
    and prepares data for MedVec insertion.
    """
    
    def __init__(self, medvec_api: Callable[[List[MedChunk]], None]):
        """Initializes with the MedVec API insertion function."""
        self.medvec_api = medvec_api

    def ingest_file(self, file_path: str, modality: str, metadata_params: Dict[str, Any] = {}) -> IngestionResult:
        """
        End-to-end processing of a single file.
        1. Process (Chunking/Metadata) -> 2. Encode -> 3. Ingest (to MedVec).
        """
        if modality not in PROCESSORS:
            return IngestionResult(chunks=[], success=False, error=f"Modality '{modality}' not supported.")

        try:
            # 1. Chunking/Processing
            processor = PROCESSORS[modality]
            chunks: List[MedChunk] = processor.process(file_path, **metadata_params)
            
            # 2. Encoding
            for chunk in chunks:
                # Apply Text Encoder if applicable
                if chunk.modality == "text" and "text_vec" in ENCODERS:
                    ENCODERS["text_vec"].encode(chunk)
                
                # Apply Multimodal Encoder (CLIP)
                if "multimodal" in ENCODERS:
                   ENCODERS["multimodal"].encode(chunk)
            
            # 3. Ingestion to MedVec
            if chunks:
                self.medvec_api(chunks)
            
            return IngestionResult(chunks=chunks, success=True)
        
        except Exception as e:
            return IngestionResult(chunks=[], success=False, error=str(e))

    # --- Public Helper Methods ---
    
    def generate_patient_vec(self, patient_id: str) -> List[float]:
        """Placeholder: Query MedVec for all chunks, aggregate, and return a single vector."""
        # TODO: Implement complex aggregation/fusion logic here (Phase 5)
        raise NotImplementedError("Patient Vector Aggregation requires MedVec retrieval.")