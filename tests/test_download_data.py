import unittest
from unittest.mock import patch, MagicMock
import sys

# Mock Kaggle API and dotenv globally to simulate restricted environments
mock_kaggle_module = MagicMock()
mock_api_class = MagicMock()
mock_kaggle_module.api.kaggle_api_extended.KaggleApi = mock_api_class
sys.modules['kaggle.api.kaggle_api_extended'] = mock_kaggle_module.api.kaggle_api_extended

sys.modules['dotenv'] = MagicMock()

from ingestion.download_data import download_kaggle_dataset

class TestDownloadData(unittest.TestCase):
    def setUp(self):
        self.mock_api_instance = mock_api_class.return_value
        self.mock_api_instance.authenticate.reset_mock()
        self.mock_api_instance.dataset_download_files.reset_mock()
        mock_api_class.reset_mock()

    @patch('os.path.exists', return_value=True)
    def test_download_with_provided_api(self, mock_exists):
        mock_provided_api = MagicMock()
        download_kaggle_dataset("test/dataset", "data", api=mock_provided_api)

        # Verify the provided API was used and NOT authenticated again
        mock_provided_api.authenticate.assert_not_called()
        mock_provided_api.dataset_download_files.assert_called_once_with("test/dataset", path="data", unzip=True)

        # Verify the globally mocked API class was NOT instantiated or authenticated
        mock_api_class.assert_not_called()

    @patch('os.path.exists', return_value=True)
    def test_download_without_provided_api(self, mock_exists):
        download_kaggle_dataset("test/dataset", "data")

        # Verify a new API instance was created, authenticated, and used
        mock_api_class.assert_called_once()
        self.mock_api_instance.authenticate.assert_called_once()
        self.mock_api_instance.dataset_download_files.assert_called_once_with("test/dataset", path="data", unzip=True)

if __name__ == '__main__':
    unittest.main()
