from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class DocumentType(str, Enum):
    PDF = "pdf"
    WORD = "docx"
    TXT = "txt"
    SOP = "sop"
    MANUAL = "manual"

class DocumentChunk(BaseModel):
    id: str
    document_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    chunk_index: int
    total_chunks: int

class Document(BaseModel):
    id: str
    title: str
    file_path: str
    doc_type: DocumentType
    created_at: datetime
    total_chunks: int
    metadata: Dict[str, Any]

class QueryContext(BaseModel):
    query: str
    intent: str
    entities: List[str]
    keywords: List[str]
    original_query: str

class RetrievedChunk(BaseModel):
    chunk: DocumentChunk
    similarity_score: float
    source: str
    verified: bool = False

class SearchResult(BaseModel):
    query: str
    chunks: List[RetrievedChunk]
    total_results: int
    processing_time_ms: float

from typing import Union

class Citation(BaseModel):
    text: str
    source: str
    similarity: float

class AnswerResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    confidence_score: float
    grounded: bool
    verification_notes: List[str]
    source_chunks: List[RetrievedChunk]

class IngestionResult(BaseModel):
    document_id: str
    success: bool
    chunks_created: int
    error: Optional[str] = None
    processing_time_ms: float