import os
import pytest
import sys
from unittest.mock import patch, MagicMock

# Mock external dependencies before importing local modules
sys.modules['kaggle'] = MagicMock()
sys.modules['kaggle.api.kaggle_api_extended'] = MagicMock()
sys.modules['dotenv'] = MagicMock()

from ingestion.download_data import download_kaggle_dataset

@patch('ingestion.download_data.KaggleApi')
@patch('ingestion.download_data.os.path.exists')
@patch('ingestion.download_data.os.makedirs')
def test_download_kaggle_dataset_success(mock_makedirs, mock_exists, MockKaggleApi, capsys):
    mock_exists.return_value = False
    mock_api_instance = MockKaggleApi.return_value

    download_kaggle_dataset('test/dataset', 'test_path')

    mock_makedirs.assert_called_once_with('test_path')
    mock_api_instance.authenticate.assert_called_once()
    mock_api_instance.dataset_download_files.assert_called_once_with('test/dataset', path='test_path', unzip=True)

    captured = capsys.readouterr()
    assert "Successfully downloaded test/dataset." in captured.out

@patch('ingestion.download_data.KaggleApi')
@patch('ingestion.download_data.os.path.exists')
def test_download_kaggle_dataset_oserror(mock_exists, MockKaggleApi, capsys):
    mock_exists.return_value = True
    mock_api_instance = MockKaggleApi.return_value
    mock_api_instance.dataset_download_files.side_effect = OSError("Disk full")

    download_kaggle_dataset('test/dataset', 'test_path')

    captured = capsys.readouterr()
    assert "Failed to download test/dataset due to OS error: Disk full" in captured.out

@patch('ingestion.download_data.KaggleApi')
@patch('ingestion.download_data.os.path.exists')
def test_download_kaggle_dataset_exception(mock_exists, MockKaggleApi, capsys):
    mock_exists.return_value = True
    mock_api_instance = MockKaggleApi.return_value
    mock_api_instance.dataset_download_files.side_effect = Exception("API rate limit")

    download_kaggle_dataset('test/dataset', 'test_path')

    captured = capsys.readouterr()
    assert "Failed to download test/dataset due to an unexpected error: API rate limit" in captured.out
