# Query understanding
from openai import OpenAI
from pydantic import BaseModel
from typing import List
import json
import re
from config import config
from logger import logger

class QueryAnalysis(BaseModel):
    intent: str
    entities: List[str]
    keywords: List[str]
    refined_query: str

class QueryAgent:
    """Analyzes and refines user queries."""
    
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.LLM_MODEL
        logger.info("QueryAgent initialized")
    
    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze and refine the user query."""
        logger.info(f"Analyzing query: {query[:50]}...")
        
        system_prompt = """You are a query analysis expert. Given a user query, extract:
1. Intent: What is the user trying to do? (e.g., troubleshoot, explain, compare, list)
2. Entities: Specific names, products, terms mentioned
3. Keywords: Important search terms
4. Refined query: A cleaner, more searchable version

Return JSON only with these fields."""
        
        user_prompt = f"Analyze this query: {query}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return QueryAnalysis(
                intent=result.get('intent', 'unknown'),
                entities=result.get('entities', []),
                keywords=result.get('keywords', []),
                refined_query=result.get('refined_query', query)
            )
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return QueryAnalysis(
                intent='unknown',
                entities=[],
                keywords=[],
                refined_query=query
            )

# Global instance
query_agent = QueryAgent()