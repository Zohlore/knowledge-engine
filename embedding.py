# Embedding generation
import numpy as np
from typing import List, Optional
from openai import OpenAI
from config import config
from logger import logger

class EmbeddingGenerator:
    """Generate embeddings for text chunks."""
    
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.EMBEDDING_MODEL
        logger.info(f"EmbeddingGenerator initialized: {self.model}")
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding for text length: {len(text)}")
            return embedding
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None
    
    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            embeddings = [data.embedding for data in response.data]
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            return [None] * len(texts)
    
    def get_embedding_size(self) -> int:
        """Get the size of embeddings."""
        try:
            test_embedding = self.embed_text("test")
            return len(test_embedding) if test_embedding else 1536
        except:
            return 1536  # Default for ada-002

# Global embedding generator
embedder = EmbeddingGenerator()