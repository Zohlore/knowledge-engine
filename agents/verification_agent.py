# Grounding + citations
from typing import List, Dict, Any
from openai import OpenAI
import json
import re
from config import config
from logger import logger
from models import RetrievedChunk

class VerificationAgent:
    """Verifies answers and adds citations."""
    
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.LLM_MODEL
        logger.info("VerificationAgent initialized")
    
    def verify(self, query: str, answer: str, chunks: List[RetrievedChunk]) -> Dict[str, Any]:
        """Verify the answer and extract citations."""
        logger.info(f"Verifying answer for: {query[:50]}...")
        
        # Build context from chunks
        context = []
        for i, chunk in enumerate(chunks, 1):
            context.append(f"[{i}] {chunk.chunk.content[:500]}...")
        
        context_str = "\n\n".join(context)
        
        system_prompt = """You are a verification expert. Given a query, an answer, and source chunks:

1. Check if the answer is grounded in the sources (yes/no)
2. Identify which sources support the answer
3. Extract specific citations
4. Note any verification issues

Return JSON with:
- grounded: boolean
- citations: list of {"source_index": int, "text": str}
- verification_notes: list of strings
- confidence_score: float (0-1)"""
        
        user_prompt = f"""Query: {query}

Answer: {answer}

Sources:
{context_str}

Verify and return JSON."""
        
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
            
            return {
                'grounded': result.get('grounded', False),
                'citations': result.get('citations', []),
                'verification_notes': result.get('verification_notes', []),
                'confidence_score': result.get('confidence_score', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {
                'grounded': False,
                'citations': [],
                'verification_notes': ['Verification service failed'],
                'confidence_score': 0.0
            }

# Global instance
verification_agent = VerificationAgent()