import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Mock kaggle dependencies before importing download_data
mock_kaggle = MagicMock()
mock_kaggle_api = MagicMock()

with patch.dict('sys.modules', {
    'kaggle': mock_kaggle,
    'kaggle.api.kaggle_api_extended': mock_kaggle_api
}):
    from ingestion.download_data import download_kaggle_dataset
    import ingestion.download_data

class TestDownloadData(unittest.TestCase):

    def setUp(self):
        mock_kaggle_api.KaggleApi.reset_mock()

    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_download_kaggle_dataset_without_api(self, mock_makedirs, mock_exists):
        # Setup mocks
        mock_exists.return_value = True
        mock_api_instance = MagicMock()
        mock_kaggle_api.KaggleApi.return_value = mock_api_instance

        # Call function
        download_kaggle_dataset('test/dataset')

        # Verify new api was instantiated and authenticated
        mock_kaggle_api.KaggleApi.assert_called_once()
        mock_api_instance.authenticate.assert_called_once()
        mock_api_instance.dataset_download_files.assert_called_once_with('test/dataset', path='data', unzip=True)

    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_download_kaggle_dataset_with_injected_api(self, mock_makedirs, mock_exists):
        # Setup mocks
        mock_exists.return_value = True
        mock_api_instance = MagicMock()

        # Call function with injected api
        download_kaggle_dataset('test/dataset', api=mock_api_instance)

        # Verify injected api was used and NOT re-authenticated
        mock_api_instance.authenticate.assert_not_called()
        mock_api_instance.dataset_download_files.assert_called_once_with('test/dataset', path='data', unzip=True)

if __name__ == '__main__':
    unittest.main()
