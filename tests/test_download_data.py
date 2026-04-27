import unittest
from unittest.mock import patch, MagicMock
import sys
import runpy

# Mock required dependencies to allow import in restricted environments
mock_kaggle = MagicMock()
mock_kaggle_api = MagicMock()
sys.modules['kaggle'] = mock_kaggle
sys.modules['kaggle.api.kaggle_api_extended'] = mock_kaggle_api
sys.modules['dotenv'] = MagicMock()

from ingestion.download_data import download_kaggle_dataset

class TestDownloadData(unittest.TestCase):
    @patch('ingestion.download_data.KaggleApi')
    @patch('os.makedirs')
    @patch('os.path.exists', return_value=False)
    def test_download_dataset_without_injected_api(self, mock_exists, mock_makedirs, mock_kaggle_api_class):
        mock_api_instance = MagicMock()
        mock_kaggle_api_class.return_value = mock_api_instance

        download_kaggle_dataset("test/dataset")

        mock_kaggle_api_class.assert_called_once()
        mock_api_instance.authenticate.assert_called_once()
        mock_api_instance.dataset_download_files.assert_called_once_with("test/dataset", path="data", unzip=True)

    @patch('ingestion.download_data.KaggleApi')
    @patch('os.makedirs')
    @patch('os.path.exists', return_value=True)
    def test_download_dataset_with_injected_api(self, mock_exists, mock_makedirs, mock_kaggle_api_class):
        mock_injected_api = MagicMock()

        download_kaggle_dataset("test/dataset", api=mock_injected_api)

        # Verify KaggleApi was not instantiated internally
        mock_kaggle_api_class.assert_not_called()

        # Verify the injected API was not re-authenticated
        mock_injected_api.authenticate.assert_not_called()

        # Verify the download was called on the injected API
        mock_injected_api.dataset_download_files.assert_called_once_with("test/dataset", path="data", unzip=True)

if __name__ == '__main__':
    unittest.main()
