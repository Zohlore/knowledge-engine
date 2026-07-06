import os
import streamlit as st

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") or st.secrets.get("TAVILY_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL") or st.secrets.get("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or st.secrets.get("QDRANT_API_KEY")

    # Collection settings
    COLLECTION_NAME = "knowledge_base"
    VECTOR_SIZE = 1536
    DISTANCE_METRIC = "Cosine"

    # Chunking settings
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50

    # Retrieval settings
    TOP_K = 5
    SIMILARITY_THRESHOLD = 0.7

    # Model settings
    EMBEDDING_MODEL = "text-embedding-ada-002"
    LLM_MODEL = "gpt-4o-mini"
    TEMPERATURE = 0.1

    # Paths
    DOCUMENTS_DIR = "data/documents"
    METADATA_DIR = "data/metadata"

# Create an instance to import
config = Config()