import pytest
from unittest.mock import patch, MagicMock
import sys

# Mock langchain modules before importing AIClientFactory
sys.modules['langchain_groq'] = MagicMock()
sys.modules['langchain_huggingface'] = MagicMock()
sys.modules['dotenv'] = MagicMock()

from core.groq_client import AIClientFactory

def test_get_groq_llm_missing_api_key():
    """Test that get_groq_llm raises ValueError when GROQ_API_KEY is missing."""
    with patch("os.getenv") as mock_getenv:
        mock_getenv.return_value = None
        with pytest.raises(ValueError) as excinfo:
            AIClientFactory.get_groq_llm()
        assert "GROQ_API_KEY not found in environment variables." in str(excinfo.value)

def test_get_groq_llm_success():
    """Test that get_groq_llm returns a ChatGroq instance when API key is present."""
    with patch("os.getenv") as mock_getenv:
        mock_getenv.return_value = "fake_api_key"
        with patch("core.groq_client.ChatGroq") as mock_chat_groq:
            llm = AIClientFactory.get_groq_llm(model_name="test-model", temperature=0.5)

            mock_chat_groq.assert_called_once_with(
                api_key="fake_api_key",
                model="test-model",
                temperature=0.5
            )
            assert llm == mock_chat_groq.return_value
