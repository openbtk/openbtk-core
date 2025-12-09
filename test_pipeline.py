import os
import sys
from typing import List

# Ensure we can import the local package
# sys.path.append(os.getcwd())

from openbtk.core.ingester import OpenBTKIngester
from openbtk.models import MedChunk

def mock_medvec_api(chunks: List[MedChunk]):
    print(f"✅ MedVec API received {len(chunks)} chunks.")
    for chunk in chunks:
        print(f"   - Chunk ID: {chunk.id}, Modality: {chunk.modality}")
        if chunk.vectors:
            print(f"     - Vectors: {list(chunk.vectors.keys())}")
        else:
            print(f"     - Vectors: None (Expected for dry run or if encoder failed)")

def main():
    print("🚀 Starting OpenBTK Pipeline Test...")

    # 1. Create dummy text file
    dummy_text = "Patient presented with mild chest pain. vital signs stable. no acute distress."
    with open("test_clinical_note.txt", "w") as f:
        f.write(dummy_text)

    # 2. Initialize Ingester
    ingester = OpenBTKIngester(medvec_api=mock_medvec_api)

    # 3. Test Text Ingestion
    print("\n------------------------------")
    print("Testing Text Ingestion...")
    result = ingester.ingest_file(
        file_path="test_clinical_note.txt", 
        modality="text", 
        metadata_params={"patient_id": "123", "encounter_id": "456"}
    )

    if result.success:
        print("✅ Text Ingestion Successful")
    else:
        print(f"❌ Text Ingestion Failed: {result.error}")

    # 4. Cleanup
    if os.path.exists("test_clinical_note.txt"):
        os.remove("test_clinical_note.txt")

    print("\n------------------------------")
    print("Test Complete.")

if __name__ == "__main__":
    main()
