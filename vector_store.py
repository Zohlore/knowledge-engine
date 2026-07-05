import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
from config import config
from logger import logger
from models import DocumentChunk, RetrievedChunk

class VectorStore:
    def __init__(self):
        self.client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            timeout=30
        )
        self.collection_name = config.COLLECTION_NAME
        self.vector_size = config.VECTOR_SIZE
        self._init_collection()
        logger.info(f"VectorStore initialized: {self.collection_name}")
    
    def _init_collection(self):
        try:
            info = self.client.get_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} exists with {info.points_count} points")
        except Exception:
            logger.info(f"Creating collection {self.collection_name} with vector size {self.vector_size}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_id",
                field_type="keyword"
            )
    
    def upsert_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> bool:
        if not chunks or not embeddings:
            logger.warning("No chunks or embeddings to upsert")
            return False
        
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point_id = str(uuid.uuid4())
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    'document_id': chunk['document_id'],
                    'content': chunk['content'],
                    'chunk_index': chunk['chunk_index'],
                    'metadata': chunk.get('metadata', {}),
                    'total_chunks': chunk.get('total_chunks', 1),
                    'chunk_id': chunk['id']
                }
            )
            points.append(point)
        
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Upserted {len(points)} chunks")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert chunks: {e}")
            return False
    
    def search(self, query_vector: List[float], top_k: int = 5,
               filter_doc_id: Optional[str] = None) -> List[RetrievedChunk]:
        filter_condition = None
        if filter_doc_id:
            filter_condition = Filter(
                must=[FieldCondition(
                    key="document_id",
                    match=MatchValue(value=filter_doc_id)
                )]
            )
        
        try:
            # Use query_points with query_vector (recommended)
            if hasattr(self.client, 'query_points'):
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                    query_filter=filter_condition,
                    with_payload=True
                ).points
            elif hasattr(self.client, 'search'):
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    query_filter=filter_condition,
                    with_payload=True
                )
            else:
                # Fallback: try query with query_vector
                results = self.client.query(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    query_filter=filter_condition,
                    with_payload=True
                )
            
            retrieved = []
            for result in results:
                chunk = DocumentChunk(
                    id=result.payload.get('chunk_id', str(result.id)),
                    document_id=result.payload.get('document_id', ''),
                    content=result.payload.get('content', ''),
                    metadata=result.payload.get('metadata', {}),
                    embedding=None,
                    chunk_index=result.payload.get('chunk_index', 0),
                    total_chunks=result.payload.get('total_chunks', 1)
                )
                retrieved.append(RetrievedChunk(
                    chunk=chunk,
                    similarity_score=result.score,
                    source=result.payload.get('metadata', {}).get('source', 'unknown')
                ))
            logger.info(f"Found {len(retrieved)} results")
            return retrieved
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        try:
            filter_condition = Filter(
                must=[FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
                )]
            )
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=filter_condition
            )
            logger.info(f"Deleted document: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        try:
            collection_info = self.client.get_collection(self.collection_name)
            points_count = getattr(collection_info, 'points_count', 0)
            return {'points_count': points_count, 'status': 'active'}
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'status': 'error', 'error': str(e)}

vector_store = VectorStore()