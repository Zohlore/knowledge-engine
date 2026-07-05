# Semantic chunking
import hashlib
import re
from typing import List, Dict, Any
from datetime import datetime
from config import config
from logger import logger

class SemanticChunker:
    """
    Semantic chunking with overlap for better context retention.
    Supports multiple document types.
    """
    
    def __init__(self):
        self.chunk_size = config.CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP
        logger.info(f"Chunker initialized: size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def chunk_pdf_text(self, text: str, document_id: str) -> List[Dict[str, Any]]:
        """Chunk PDF text with semantic boundaries."""
        logger.info(f"Chunking PDF: {document_id}")
        
        # Clean text
        text = self._clean_text(text)
        
        # Split into paragraphs first
        paragraphs = self._split_paragraphs(text)
        
        # Create chunks from paragraphs
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            if current_size + para_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, document_id, len(chunks)))
                
                # Start new chunk with overlap
                overlap = self._get_overlap(current_chunk)
                current_chunk = overlap
                current_size = sum(len(p) for p in current_chunk)
            
            current_chunk.append(para)
            current_size += para_size
        
        # Save final chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, document_id, len(chunks)))
        
        logger.info(f"Created {len(chunks)} chunks for {document_id}")
        return chunks
    
    def chunk_structured_text(self, text: str, document_id: str, 
                               headers: List[str] = None) -> List[Dict[str, Any]]:
        """Chunk structured text by sections/headers."""
        logger.info(f"Chunking structured text: {document_id}")
        
        if not headers:
            headers = ['Section', 'Chapter', 'Part', 'Heading', 'Title']
        
        chunks = []
        current_section = []
        current_header = "Introduction"
        
        lines = text.split('\n')
        
        for line in lines:
            # Check if line is a header
            is_header = False
            for header in headers:
                if line.strip().startswith(header) or re.match(rf'^{header}\s+\d+', line, re.I):
                    if current_section:
                        chunk_text = "\n".join(current_section)
                        chunks.append(self._create_chunk(
                            chunk_text, document_id, len(chunks), 
                            metadata={'section': current_header}
                        ))
                    
                    current_header = line.strip()
                    current_section = [line]
                    is_header = True
                    break
            
            if not is_header:
                current_section.append(line)
        
        # Save final chunk
        if current_section:
            chunk_text = "\n".join(current_section)
            chunks.append(self._create_chunk(
                chunk_text, document_id, len(chunks),
                metadata={'section': current_header}
            ))
        
        logger.info(f"Created {len(chunks)} structured chunks for {document_id}")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean text for better chunking."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return text
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split by double newline or multiple spaces
        paragraphs = re.split(r'\n\s*\n|\n{2,}', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _get_overlap(self, chunk: List[str]) -> List[str]:
        """Get overlapping paragraphs for the next chunk."""
        overlap_size = 0
        overlap = []
        
        for para in reversed(chunk):
            if overlap_size + len(para) <= self.chunk_overlap:
                overlap.insert(0, para)
                overlap_size += len(para)
            else:
                break
        
        return overlap
    
    def _create_chunk(self, text: str, doc_id: str, idx: int, 
                      metadata: Dict = None) -> Dict[str, Any]:
        """Create a chunk dictionary."""
        chunk_id = hashlib.md5(f"{doc_id}_{idx}_{text[:100]}".encode()).hexdigest()[:12]
        
        metadata = metadata or {}
        metadata.update({
            'document_id': doc_id,
            'chunk_index': idx,
            'source': 'semantic_chunking'
        })
        
        return {
            'id': chunk_id,
            'document_id': doc_id,
            'content': text,
            'metadata': metadata,
            'chunk_index': idx,
            'total_chunks': 0  # Will be set after all chunks are created
        }

# Global chunker instance
chunker = SemanticChunker()