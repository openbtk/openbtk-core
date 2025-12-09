from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from ..models import MedChunk
import os

class MultimodalEncoder:
    """
    Encoder using CLIP for both image and text modalities (zero-shot capabilities).
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)

    def encode(self, chunk: MedChunk) -> None:
        """
        Encodes image or text content into the same vector space.
        """
        try:
            if chunk.modality == "image":
                image_path = chunk.raw_data_link
                if not os.path.exists(image_path):
                    return
                image = Image.open(image_path)
                inputs = self.processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model.get_image_features(**inputs)
                
                chunk.vectors["image_vec"] = outputs[0].tolist()

            elif chunk.modality == "text":
                text = chunk.text_content
                if not text:
                    return
                inputs = self.processor(text=[text], return_tensors="pt", padding=True)
                with torch.no_grad():
                    outputs = self.model.get_text_features(**inputs)
                
                chunk.vectors["text_vec_clip"] = outputs[0].tolist()
                
        except Exception as e:
            print(f"Error encoding chunk {chunk.id}: {e}")
