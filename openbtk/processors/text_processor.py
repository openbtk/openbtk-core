from typing import List, Dict, Any
from ..models import MedChunk
import uuid

class TextProcessor:
    """
    Processor for text-based biomedical data (clinical notes, reports).
    """
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def process(self, file_path: str, **kwargs) -> List[MedChunk]:
        """
        Reads a text file and splits it into chunks.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            # Fallback for different encodings if needed
            print(f"Error reading file {file_path}: {e}")
            return []

        chunks = self._chunk_text(content)
        
        med_chunks = []
        for i, text_chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            med_chunks.append(MedChunk(
                id=chunk_id,
                patient_id=kwargs.get("patient_id", "unknown"),
                encounter_id=kwargs.get("encounter_id", "unknown"),
                modality="text",
                text_content=text_chunk,
                raw_data_link=file_path,
                metadata={
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    **kwargs
                }
            ))
        return med_chunks

    def _chunk_text(self, text: str) -> List[str]:
        """
        Simple sliding window chunking. 
        TODO: Implement sentence-boundary aware chunking.
        """
        tokens = text.split() # Naive tokenization by whitespace
        chunks = []
        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunks.append(" ".join(chunk_tokens))
        
        if not chunks and text:
            chunks.append(text)
            
        return chunks
