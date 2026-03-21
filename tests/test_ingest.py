import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock pandas, langchain_chroma, and other dependencies that might not be available
sys.modules['pandas'] = MagicMock()
sys.modules['langchain_chroma'] = MagicMock()
sys.modules['core'] = MagicMock()
sys.modules['core.groq_client'] = MagicMock()
sys.modules['langchain_core'] = MagicMock()
sys.modules['langchain_core.documents'] = MagicMock()

# Import the module to test after mocking dependencies
from ingestion.ingest import ingest_bhagavad_gita
import ingestion.ingest as ingest_module

class TestIngest(unittest.TestCase):

    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('builtins.print')
    def test_ingest_document_creation(self, mock_print, mock_makedirs, mock_exists):
        # Setup mocks
        mock_exists.return_value = True

        # Mock pandas read_csv
        mock_df = MagicMock()
        ingest_module.pd.read_csv.return_value = mock_df

        # Mock the dataframe filtering
        mock_krishna_df = MagicMock()
        mock_df.__getitem__.return_value = mock_krishna_df
        mock_krishna_df.empty = False

        # Mock rows for itertuples
        class MockRow:
            def __init__(self, chapter, verse, meaning):
                self.Chapter = chapter
                self.Verse = verse
                self.EngMeaning = meaning

        mock_rows = [
            MockRow(1, 1, "Meaning 1"),
            MockRow(1, 2, "Meaning 2")
        ]
        mock_krishna_df.itertuples.return_value = mock_rows

        # Mock Document
        mock_document_class = MagicMock()
        ingest_module.Document = mock_document_class

        # Mock Chroma
        mock_chroma_class = MagicMock()
        ingest_module.Chroma = mock_chroma_class

        # Run function
        ingest_bhagavad_gita("dummy_path.csv")

        # Verify itertuples was called
        mock_krishna_df.itertuples.assert_called_once()

        # Verify Documents were created correctly
        self.assertEqual(mock_document_class.call_count, 2)

        # Check first document creation args
        args, kwargs = mock_document_class.call_args_list[0]
        self.assertEqual(kwargs['page_content'], "Meaning 1")
        self.assertEqual(kwargs['metadata']['chapter'], 1)
        self.assertEqual(kwargs['metadata']['verse'], 1)
        self.assertEqual(kwargs['metadata']['speaker'], "Krishna")

        # Check second document creation args
        args, kwargs = mock_document_class.call_args_list[1]
        self.assertEqual(kwargs['page_content'], "Meaning 2")
        self.assertEqual(kwargs['metadata']['chapter'], 1)
        self.assertEqual(kwargs['metadata']['verse'], 2)
        self.assertEqual(kwargs['metadata']['speaker'], "Krishna")

if __name__ == '__main__':
    unittest.main()
