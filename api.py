# FastAPI
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from datetime import datetime
import shutil
from pathlib import Path

from orchestrator import orchestrator
from ingest import ingestor
from logger import logger

app = FastAPI(
    title="Knowledge Engine API",
    description="Multi-Agent RAG System with Citations",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[dict]
    confidence_score: float
    grounded: bool
    verification_notes: List[str]
    processing_time_ms: float

class IngestionResponse(BaseModel):
    success: bool
    document_id: str
    chunks_created: int
    message: str
    processing_time_ms: float

@app.get("/")
async def root():
    return {"status": "healthy", "service": "Knowledge Engine"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the knowledge base."""
    logger.info(f"API query: {request.query[:50]}...")
    start_time = datetime.now()
    
    try:
        response = orchestrator.query(request.query, top_k=request.top_k)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return QueryResponse(
            query=response.query,
            answer=response.answer,
            citations=response.citations,
            confidence_score=response.confidence_score,
            grounded=response.grounded,
            verification_notes=response.verification_notes,
            processing_time_ms=processing_time
        )
    except Exception as e:
        logger.error(f"Query API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest", response_model=IngestionResponse)
async def ingest_document(file: UploadFile = File(...)):
    """Ingest a PDF document."""
    logger.info(f"API ingest: {file.filename}")
    start_time = datetime.now()
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save file
        save_path = Path("data/documents") / file.filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        
        # Ingest
        result = ingestor.ingest_pdf(str(save_path))
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return IngestionResponse(
            success=result.success,
            document_id=result.document_id,
            chunks_created=result.chunks_created,
            message="Ingestion successful" if result.success else result.error,
            processing_time_ms=processing_time
        )
    except Exception as e:
        logger.error(f"Ingest API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)