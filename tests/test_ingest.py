import pytest
from unittest.mock import patch, MagicMock
import sys
import pandas as pd

# Mock langchain and core modules before importing ingest.py
sys.modules['langchain_core.documents'] = MagicMock()
sys.modules['langchain_core'] = MagicMock()
sys.modules['langchain_chroma'] = MagicMock()
sys.modules['core.groq_client'] = MagicMock()

from ingestion.ingest import ingest_bhagavad_gita

def test_ingest_bhagavad_gita_itertuples(tmp_path):
    """Test that ingest_bhagavad_gita runs without errors with the new itertuples logic."""
    # Create a dummy CSV file
    csv_file = tmp_path / "dummy_gita.csv"
    data = {
        "Chapter": [1, 2],
        "Verse": [1, 2],
        "EngMeaning": ["Meaning 1", "Meaning 2"],
        "Transliteration": ["śrī-bhagavān uvāca", "śrī-bhagavān uvāca"]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

    # Mock the Chroma and embeddings client
    with patch("ingestion.ingest.Chroma") as mock_chroma, \
         patch("ingestion.ingest.AIClientFactory.get_huggingface_embeddings") as mock_get_embeddings, \
         patch("ingestion.ingest.Document") as mock_document:

        vector_store_path = str(tmp_path / "vector_store")

        # Call the function
        ingest_bhagavad_gita(str(csv_file), vector_store_path)

        # Verify Document was called correctly for each row
        assert mock_document.call_count == 2
        mock_document.assert_any_call(page_content="Meaning 1", metadata={"chapter": 1, "verse": 1, "speaker": "Krishna"})
        mock_document.assert_any_call(page_content="Meaning 2", metadata={"chapter": 2, "verse": 2, "speaker": "Krishna"})

        # Verify Chroma was called to create the vector store
        mock_chroma.from_documents.assert_called_once()
