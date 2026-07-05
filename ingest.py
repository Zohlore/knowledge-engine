import os
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF

from config import config
from logger import logger
from models import IngestionResult
from chunking import chunker
from embedding import embedder
from vector_store import vector_store

class DocumentIngestor:
    """Ingest documents into the knowledge base."""
    
    def __init__(self):
        self.documents_dir = Path(config.DOCUMENTS_DIR)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ingestor initialized: {self.documents_dir}")
    
    def ingest_pdf(self, file_path: str) -> IngestionResult:
        """Ingest a PDF document."""
        logger.info(f"Ingesting PDF: {file_path}")
        start_time = time.time()
        
        try:
            # Extract text from PDF
            text = self._extract_pdf_text(file_path)
            if not text:
                return IngestionResult(
                    document_id="",
                    success=False,
                    chunks_created=0,
                    error="No text extracted from PDF",
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            
            # Generate document ID
            doc_id = hashlib.md5(file_path.encode()).hexdigest()[:12]
            
            # Chunk the text
            chunks = chunker.chunk_pdf_text(text, doc_id)
            
            if not chunks:
                return IngestionResult(
                    document_id=doc_id,
                    success=False,
                    chunks_created=0,
                    error="No chunks created",
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            
            # Set total chunks for each chunk
            for chunk in chunks:
                chunk['total_chunks'] = len(chunks)
            
            # Generate embeddings for all chunks
            chunk_texts = [c['content'] for c in chunks]
            embeddings = embedder.embed_batch(chunk_texts)
            
            # Filter out failed embeddings
            valid_chunks = []
            valid_embeddings = []
            for chunk, emb in zip(chunks, embeddings):
                if emb is not None:
                    valid_chunks.append(chunk)
                    valid_embeddings.append(emb)
                else:
                    logger.warning(f"Skipping chunk {chunk['id']} due to embedding failure")
            
            if not valid_chunks:
                return IngestionResult(
                    document_id=doc_id,
                    success=False,
                    chunks_created=0,
                    error="No valid embeddings generated",
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            
            # Store in vector database
            success = vector_store.upsert_chunks(valid_chunks, valid_embeddings)
            
            if success:
                logger.info(f"✅ Successfully ingested {file_path}")
                return IngestionResult(
                    document_id=doc_id,
                    success=True,
                    chunks_created=len(valid_chunks),
                    error=None,
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            else:
                return IngestionResult(
                    document_id=doc_id,
                    success=False,
                    chunks_created=0,
                    error="Failed to store in vector database",
                    processing_time_ms=(time.time() - start_time) * 1000
                )
                
        except Exception as e:
            logger.error(f"PDF ingestion failed: {e}")
            return IngestionResult(
                document_id="",
                success=False,
                chunks_created=0,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF using PyMuPDF."""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return ""
    
    def ingest_directory(self, directory_path: str) -> List[IngestionResult]:
        """Ingest all documents in a directory."""
        logger.info(f"Ingesting directory: {directory_path}")
        results = []
        
        for file in Path(directory_path).iterdir():
            if file.suffix.lower() == '.pdf':
                result = self.ingest_pdf(str(file))
                results.append(result)
            elif file.suffix.lower() in ['.docx', '.txt', '.md']:
                # Add support for other formats
                result = self.ingest_generic(str(file))
                results.append(result)
        
        logger.info(f"Ingested {len(results)} documents")
        return results
    
    def ingest_generic(self, file_path: str) -> IngestionResult:
        """Ingest any document using Unstructured."""
        logger.info(f"Ingesting generic: {file_path}")
        # Simplified - you can expand this
        return IngestionResult(
            document_id="",
            success=False,
            chunks_created=0,
            error="Generic ingestion not implemented",
            processing_time_ms=0
        )

# Global ingestor
ingestor = DocumentIngestor()