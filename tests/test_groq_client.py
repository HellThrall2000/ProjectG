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

def test_get_huggingface_embeddings_success():
    """Test that get_huggingface_embeddings returns a HuggingFaceEmbeddings instance."""
    AIClientFactory.get_huggingface_embeddings.cache_clear()

    with patch("core.groq_client.HuggingFaceEmbeddings") as mock_hf_embeddings:
        # Test default model name
        embeddings_default = AIClientFactory.get_huggingface_embeddings()
        mock_hf_embeddings.assert_called_once_with(model_name="all-MiniLM-L6-v2")
        assert embeddings_default == mock_hf_embeddings.return_value

        # Clear cache and test custom model name
        AIClientFactory.get_huggingface_embeddings.cache_clear()
        mock_hf_embeddings.reset_mock()

        embeddings_custom = AIClientFactory.get_huggingface_embeddings(model_name="custom-model")
        mock_hf_embeddings.assert_called_once_with(model_name="custom-model")
        assert embeddings_custom == mock_hf_embeddings.return_value
