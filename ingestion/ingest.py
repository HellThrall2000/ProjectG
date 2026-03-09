import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma  # Updated import for Chroma vector store
from core.groq_client import AIClientFactory

# ...existing code...

def ingest_bhagavad_gita(csv_path: str, vector_store_path: str = "./vector_store/philosophical"):
    """
    Ingests data from the Bhagwad_Gita.csv file, extracts the speaker,
    filters for Krishna's speech, and stores it in a ChromaDB vector store.

    Args:
        csv_path (str): The path to the Bhagwad_Gita.csv file.
        vector_store_path (str): The path to the vector store directory.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    df = pd.read_csv(csv_path)

    def get_speaker(transliteration):
        if "uvāca" in transliteration:
            return transliteration.split(" uvāca")[0].strip()
        return "Unknown"

    df["Speaker"] = df["Transliteration"].apply(get_speaker)

    # Filter for Krishna's speech (śrī-bhagavān uvāca)
    krishna_df = df[df["Speaker"] == "śrī-bhagavān"]

    if krishna_df.empty:
        print("No speech from Krishna found in the provided CSV file.")
        return

    # Create documents
    documents = []
    for _, row in krishna_df.iterrows():
        metadata = {
            "chapter": row["Chapter"],
            "verse": row["Verse"],
            "speaker": "Krishna"
        }
        document = Document(page_content=row["EngMeaning"], metadata=metadata)
        documents.append(document)

    # Get embeddings client
    embeddings = AIClientFactory.get_huggingface_embeddings()

    # Create vector store
    if not os.path.exists(vector_store_path):
        os.makedirs(vector_store_path)

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=vector_store_path
    )
    print(f"Data ingested successfully into {vector_store_path}")

if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ingest_bhagavad_gita(os.path.join(project_root, "data/Bhagwad_Gita.csv"))