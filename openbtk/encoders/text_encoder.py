# Dependencies: transformers (for BioBERT), torch/tensorflow
from transformers import AutoTokenizer, AutoModel
import torch
from ..models import MedChunk

class ClinicalTextEncoder:
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, chunk: MedChunk) -> None:
        """Encodes the text_content and populates the text_vec field."""
        if not chunk.text_content:
            return

        # Tokenize and get embedding
        inputs = self.tokenizer(chunk.text_content, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad(): # Assuming PyTorch
            outputs = self.model(**inputs)
        
        # Use CLS token output as the chunk vector
        embedding = outputs.last_hidden_state[0, 0, :].tolist()
        
        # Populate the MedChunk object
        chunk.vectors["text_vec"] = embedding