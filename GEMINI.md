# Project Context: Krishna AI Persona

## Overview
This repository contains a Retrieval-Augmented Generation (RAG) system designed to emulate the persona of Krishna. The AI segregates the persona into three distinct functional aspects:
1. **Philosophical** (Source: Bhagavad Gita)
2. **Diplomatic** (Source: Mahabharata)
3. **Personal/Divine** (Source: Srimad Bhagavatam)

## Tech Stack & Engine
* **Orchestration:** LangGraph (Multi-agent routing and supervisor architecture)
* **Inference / LLM:** Groq (`langchain-groq`) for high-speed generation.
* **Embeddings:** Local HuggingFace embeddings (`all-MiniLM-L6-v2`) via `langchain-huggingface`.
* **Vector Database:** ChromaDB.
* **Data Ingestion:** `CSVLoader` (Strictly structured `.csv` data, avoiding raw `.txt` files).

## Architecture & Design Patterns
* **Factory Pattern:** AI clients (Groq LLM, HuggingFace Embeddings) must be instantiated using the `AIClientFactory` in `core/groq_client.py` to decouple configuration from execution.
* **Modular Design:** Keep ingestion, retrieval, and agent routing in strictly separated modules (`ingestion/`, `retrieval/`, `agents/`).
* **Metadata Enforcement:** All ingested documents must map row data to specific metadata (e.g., `chapter`, `verse`, `speaker`).
* **Speaker Filtering:** The ingestion pipeline must actively filter data to ensure only rows where `Speaker == "Krishna"` (or equivalent variations) are embedded into the vector store. Do not embed narrator or interlocutor text (e.g., Sanjaya, Arjuna).

## Coding Standards
1. **Modern Python:** Use Python 3.10+ features.
2. **Type Hinting:** Enforce strict type hinting on all function signatures and class methods.
3. **Docstrings:** Use Google-style docstrings for all classes and complex functions.
4. **Environment Variables:** Never hardcode API keys. Always use `python-dotenv` and retrieve keys like `GROQ_API_KEY` via `os.getenv()`.
5. **Clean Code:** Prioritize readability and maintainability. Avoid deeply nested loops; use list comprehensions or generators where appropriate.

## Agent Behavior (LangGraph)
* The system relies on a Supervisor Agent to interpret the user's prompt and route it to the correct specialized vector retriever (Philosophical, Diplomatic, or Personal) based on semantic intent.
* Generated responses must ground themselves strictly in the retrieved CSV context, avoiding general, pre-trained LLM hallucinations about the domain.s