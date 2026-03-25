import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Mock dependencies not installed in the test environment
mock_pandas = MagicMock()
mock_langchain_core_documents = MagicMock()
mock_langchain_chroma = MagicMock()
mock_core_groq_client = MagicMock()

sys.modules['pandas'] = mock_pandas
sys.modules['langchain_core'] = MagicMock()
sys.modules['langchain_core.documents'] = mock_langchain_core_documents
sys.modules['langchain_chroma'] = mock_langchain_chroma
sys.modules['core'] = MagicMock()
sys.modules['core.groq_client'] = mock_core_groq_client

from ingestion.ingest import ingest_bhagavad_gita

class TestIngest(unittest.TestCase):
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_ingest_bhagavad_gita_optimized_iteration(self, mock_makedirs, mock_exists):
        # Setup mocks
        mock_exists.return_value = True # Pretend file exists

        # Create a mock dataframe that simulates the output of pd.read_csv
        mock_df = MagicMock()
        mock_pandas.read_csv.return_value = mock_df

        # Simulate df['Speaker'] assignment and get_speaker apply
        mock_df_dict = {}
        def mock_setitem(key, val):
            mock_df_dict[key] = val
        mock_df.__setitem__.side_effect = mock_setitem

        # When we filter krishna_df = df[df["Speaker"] == "śrī-bhagavān"]
        mock_krishna_df = MagicMock()
        mock_krishna_df.empty = False

        # Create mock namedtuples to yield from itertuples
        class MockRow:
            def __init__(self, chapter, verse, meaning):
                self.Chapter = chapter
                self.Verse = verse
                self.EngMeaning = meaning

        mock_krishna_df.itertuples.return_value = [
            MockRow("1", "1", "First verse meaning"),
            MockRow("1", "2", "Second verse meaning")
        ]

        # MagicMock __getitem__ to return mock_krishna_df when filtered
        mock_df.__getitem__.return_value = mock_krishna_df

        # Run the function
        ingest_bhagavad_gita("dummy_path.csv", "dummy_vector_store")

        # Verify itertuples was called on the filtered dataframe
        mock_krishna_df.itertuples.assert_called_once_with(index=False)

        # Verify Document was created twice
        self.assertEqual(mock_langchain_core_documents.Document.call_count, 2)

        # Verify Document called with correct kwargs for the first row
        mock_langchain_core_documents.Document.assert_any_call(
            page_content="First verse meaning",
            metadata={'chapter': "1", 'verse': "1", 'speaker': 'Krishna'}
        )

        # Verify Document called with correct kwargs for the second row
        mock_langchain_core_documents.Document.assert_any_call(
            page_content="Second verse meaning",
            metadata={'chapter': "1", 'verse': "2", 'speaker': 'Krishna'}
        )

if __name__ == '__main__':
    unittest.main()
