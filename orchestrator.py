# Agent orchestration
import time
from typing import List, Dict, Any
from openai import OpenAI
from config import config
from logger import logger
from models import AnswerResponse, Citation  # <-- Added Citation import
from agents.query_agent import query_agent
from agents.retrieval_agent import retrieval_agent
from agents.verification_agent import verification_agent

class KnowledgeOrchestrator:
    """Orchestrates all agents for a complete RAG pipeline."""
    
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.LLM_MODEL
        logger.info("KnowledgeOrchestrator initialized")
    
    def query(self, question: str, top_k: int = 5) -> AnswerResponse:
        """Process a query through the entire pipeline."""
        logger.info(f"Processing query: {question[:50]}...")
        start_time = time.time()
        
        # Step 1: Analyze query
        query_analysis = query_agent.analyze(question)
        logger.info(f"Query analysis: intent={query_analysis.intent}")
        
        # Step 2: Retrieve chunks
        chunks = retrieval_agent.retrieve(query_analysis.refined_query)
        
        if not chunks:
            return AnswerResponse(
                query=question,
                answer="No relevant information found in the knowledge base.",
                citations=[],
                confidence_score=0.0,
                grounded=False,
                verification_notes=["No relevant sources found"],
                source_chunks=[]
            )
        
        # Step 3: Generate answer
        answer = self._generate_answer(question, chunks)
        
        # Step 4: Verify and add citations
        verification = verification_agent.verify(question, answer, chunks)
        
        # Build citations using the Citation model (not dictionaries)
        citations = []
        for citation_data in verification.get('citations', []):
            idx = citation_data.get('source_index', 0) - 1
            if 0 <= idx < len(chunks):
                citations.append(
                    Citation(
                        text=chunks[idx].chunk.content[:200] + "...",
                        source=chunks[idx].source,
                        similarity=chunks[idx].similarity_score
                    )
                )
        
        processing_time = (time.time() - start_time) * 1000
        
        return AnswerResponse(
            query=question,
            answer=answer,
            citations=citations,
            confidence_score=verification.get('confidence_score', 0.0),
            grounded=verification.get('grounded', False),
            verification_notes=verification.get('verification_notes', []),
            source_chunks=chunks
        )
    
    def _generate_answer(self, question: str, chunks: List) -> str:
        """Generate answer from retrieved chunks."""
        # Build context
        context = []
        for i, chunk in enumerate(chunks[:5], 1):
            context.append(f"[Source {i}]\n{chunk.chunk.content[:800]}")
        
        context_str = "\n\n".join(context)
        
        system_prompt = """You are a knowledgeable assistant. Answer the question based ONLY on the provided context.

RULES:
1. If the context doesn't contain the answer, say "I don't have enough information."
2. Cite sources using [Source X] notation.
3. Be concise and factual.
4. Do not use external knowledge."""
        
        user_prompt = f"""Context:
{context_str}

Question: {question}

Answer (citing sources):"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "Sorry, I encountered an error generating the answer."

# Global orchestrator
orchestrator = KnowledgeOrchestrator()