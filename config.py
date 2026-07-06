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
    # ... rest of config