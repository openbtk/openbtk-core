# OpenBTK (Open Biomedical ToolKit)

**OpenBTK** is an open-source library designed to streamline the ingestion, processing, and vectorization of multi-modal biomedical data. It provides a unified pipeline for handling clinical text, medical images, and biosignals (audio), making them ready for Vector Databases and LLM applications.

## 🚀 Key Features

- **📊 Multi-Modal Ingestion**: 
  - **Text**: Clinical notes, discharge summaries, reports.
  - **Image**: X-Rays, CT slices, Ultrasounds.
  - **Audio**: PCG (Phonocardiogram) and other bio-acoustic signals.
- **🧠 Domain-Specific Encoders**:
  - Integrated with **BioBERT** for clinical text embeddings.
  - **CLIP** support for joint image-text embeddings.
- **⚡ Automated Processing**: Smart chunking strategies (sliding window) and artifact validation.
- **🔌 Extensible Architecture**: Easily plug in custom processors or models.

## 📦 Installation

Install easily via pip:

```bash
pip install openbtk
```

## 🛠 Quick Start

Here is a simple example of how to ingest a clinical note.

```python
from openbtk import OpenBTKIngester

# Define a callback to handle the processed chunks (e.g., save to VectorDB)
def save_to_db(chunks):
    print(f"✅ Ready to insert {len(chunks)} chunks into VectorDB")
    for chunk in chunks:
        print(f"   - ID: {chunk.id} | Modality: {chunk.modality}")
        if "text_vec" in chunk.vectors:
             print(f"   - Vector Shape: {len(chunk.vectors['text_vec'])}")

# Initialize the Ingester
ingester = OpenBTKIngester(medvec_api=save_to_db)

# Ingest a file
result = ingester.ingest_file(
    file_path="clinical_note.txt", 
    modality="text", 
    metadata_params={"patient_id": "P123", "encounter_id": "E456"}
)

if result.success:
    print("Ingestion Complete!")
else:
    print(f"Error: {result.error}")
```

## 🏗 Architecture

OpenBTK follows a modular pipeline approach:

1.  **Ingester**: The central orchestrator that manages the flow.
2.  **Processors**: Modality-specific handlers that load raw data and split it into `MedChunks`.
3.  **Encoders**: Models that convert `MedChunks` into vector embeddings (`text_vec`, `image_vec`).
4.  **Output**: Standardized chunks ready for insertion into Pinecone, Milvus, Weaviate, or your custom solution.

## 🤝 Contributing

We welcome contributions! Please check out the issues tab to suggest features or report bugs.

## 📄 License

MIT License. See `LICENSE` for more details.
