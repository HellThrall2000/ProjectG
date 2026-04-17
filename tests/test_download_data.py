import unittest
from unittest.mock import patch, MagicMock

# Mock out missing dependencies for the restricted environment
mock_kaggle = MagicMock()
mock_kaggle_api = MagicMock()

with patch.dict('sys.modules', {
    'kaggle': mock_kaggle,
    'kaggle.api.kaggle_api_extended': mock_kaggle_api,
    'dotenv': MagicMock()
}):
    from ingestion.download_data import download_kaggle_dataset

class TestDownloadData(unittest.TestCase):
    def setUp(self):
        # Reset mocks before each test
        mock_kaggle_api.reset_mock()
        mock_kaggle.reset_mock()

    @patch('os.path.exists', return_value=True)
    def test_download_with_injected_api(self, mock_exists):
        mock_api_instance = MagicMock()

        # Call the function with the injected API
        download_kaggle_dataset('test/slug', 'test/path', api=mock_api_instance)

        # Verify the injected API was used to download files
        mock_api_instance.dataset_download_files.assert_called_once_with('test/slug', path='test/path', unzip=True)

        # Verify authenticate was NOT called on the injected API
        mock_api_instance.authenticate.assert_not_called()

        # Verify a new KaggleApi was NOT instantiated
        mock_kaggle_api.KaggleApi.assert_not_called()

    @patch('os.path.exists', return_value=True)
    def test_download_without_injected_api(self, mock_exists):
        # Call the function without providing an API
        download_kaggle_dataset('test/slug', 'test/path')

        # Verify KaggleApi was instantiated
        mock_kaggle_api.KaggleApi.assert_called_once()

        # Verify authenticate was called on the newly created instance
        created_instance = mock_kaggle_api.KaggleApi.return_value
        created_instance.authenticate.assert_called_once()

        # Verify it downloaded files
        created_instance.dataset_download_files.assert_called_once_with('test/slug', path='test/path', unzip=True)

if __name__ == '__main__':
    unittest.main()
