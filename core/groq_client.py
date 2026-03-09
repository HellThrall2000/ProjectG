import os
from functools import lru_cache
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

class AIClientFactory:
    """
    Factory class for creating AI clients.
    This class decouples the configuration of AI clients from their usage.
    """

    @staticmethod
    def get_groq_llm(model_name: str = "llama3-8b-8192", temperature: float = 0.7) -> ChatGroq:
        """
        Returns a Groq LLM client.

        Args:
            model_name (str): The name of the model to use.
            temperature (float): The temperature to use for generation.

        Returns:
            ChatGroq: A Groq LLM client.
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        return ChatGroq(
            api_key=api_key,
            model=model_name,
            temperature=temperature,
        )

    @staticmethod
    @lru_cache(maxsize=32)
    def get_huggingface_embeddings(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
        """
        Returns a HuggingFace embeddings client.

        Args:
            model_name (str): The name of the model to use.

        Returns:
            HuggingFaceEmbeddings: A HuggingFace embeddings client.
        """
        return HuggingFaceEmbeddings(model_name=model_name)

if __name__ == '__main__':
    # Example usage:
    llm = AIClientFactory.get_groq_llm()
    embeddings = AIClientFactory.get_huggingface_embeddings()
    print("LLM:", llm)
    print("Embeddings:", embeddings)
