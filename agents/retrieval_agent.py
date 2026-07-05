from typing import List
from config import config
from logger import logger
from vector_store import vector_store
from embedding import embedder
from models import RetrievedChunk

class RetrievalAgent:
    def __init__(self):
        self.top_k = config.TOP_K
        self.threshold = config.SIMILARITY_THRESHOLD
        logger.info(f"RetrievalAgent initialized: top_k={self.top_k}, threshold={self.threshold}")
    
    def retrieve(self, query: str) -> List[RetrievedChunk]:
        logger.info(f"Retrieving for: {query[:50]}...")
        query_embedding = embedder.embed_text(query)
        if query_embedding is None:
            logger.error("Failed to generate query embedding")
            return []
        results = vector_store.search(query_embedding, top_k=self.top_k)
        filtered = [r for r in results if r.similarity_score >= self.threshold]
        logger.info(f"Found {len(filtered)} relevant chunks")
        return filtered

retrieval_agent = RetrievalAgent()