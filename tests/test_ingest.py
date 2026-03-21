import pytest
import sys
from unittest.mock import MagicMock

# Create a mock dictionary for sys.modules
mock_modules = {
    'pandas': MagicMock(),
    'langchain_core': MagicMock(),
    'langchain_core.documents': MagicMock(),
    'langchain_chroma': MagicMock(),
    'core.groq_client': MagicMock()
}

# Apply the mock to sys.modules temporarily for import
import unittest.mock
with unittest.mock.patch.dict('sys.modules', mock_modules):
    from ingestion.ingest import get_speaker

def test_get_speaker_with_uvaca():
    """Test get_speaker correctly extracts speaker when 'uvāca' is present."""
    assert get_speaker("śrī-bhagavān uvāca") == "śrī-bhagavān"
    assert get_speaker("arjuna uvāca") == "arjuna"
    assert get_speaker("sañjaya uvāca") == "sañjaya"
    # Ensure it handles trailing spaces and surrounding text
    assert get_speaker("śrī-bhagavān uvāca  ") == "śrī-bhagavān"

def test_get_speaker_without_uvaca():
    """Test get_speaker returns 'Unknown' when 'uvāca' is not present."""
    assert get_speaker("some other text") == "Unknown"
    assert get_speaker("śrī-bhagavān") == "Unknown"
    assert get_speaker("uvac") == "Unknown"

def test_get_speaker_edge_cases():
    """Test get_speaker with edge cases like empty string, None, and non-strings."""
    assert get_speaker("") == "Unknown"
    assert get_speaker(" ") == "Unknown"
    # Test with non-string types as a precaution, though type hinted as str
    assert get_speaker(None) == "Unknown" # type: ignore
    assert get_speaker(123) == "Unknown" # type: ignore
    assert get_speaker(float("nan")) == "Unknown" # type: ignore
