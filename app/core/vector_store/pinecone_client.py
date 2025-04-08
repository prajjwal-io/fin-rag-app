
import logging
from typing import List, Dict, Any, Optional
import time
from pinecone import Pinecone, ServerlessSpec
import json
from ...config.settings import settings

logger = logging.getLogger(__name__)

class PineconeVectorStore:
    def __init__(self):
        self.api_key = settings.PINECONE_API_KEY
        self.environment = settings.PINECONE_ENVIRONMENT
        self.index_name = settings.PINECONE_INDEX_NAME
        self.dimension = settings.EMBEDDING_DIMENSION
        self.pc = None
        self.index = None
        self.init_connection()
    
    def init_connection(self):
        """Initialize connection to Pinecone"""
        try:
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=self.api_key)
            
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                # Create a new serverless index
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-west-2")
                )
                # Wait for index to be ready
                time.sleep(20)
            
            # Connect to the index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error connecting to Pinecone: {str(e)}")
    
    async def upsert_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> bool:
        """
        Insert or update documents in the vector store
        
        Args:
            documents: List of document dictionaries
            embeddings: List of embedding vectors
            
        Returns:
            Success status
        """
        if not self.index:
            logger.error("Pinecone index not initialized")
            return False
        
        try:
            vectors = []
            
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                # Create a unique ID
                doc_id = f"{doc.get('ticker', 'unknown')}_{doc.get('content_type', 'doc')}_{doc.get('chunk_id', i)}"
                
                # Prepare metadata
                metadata = {
                    "ticker": doc.get("ticker", ""),
                    "content_type": doc.get("content_type", "document"),
                    "source": doc.get("source", ""),
                    "chunk_id": doc.get("chunk_id", i),
                    "chunk_count": doc.get("chunk_count", 1)
                }
                
                # Add additional metadata if available
                if "filing_type" in doc:
                    metadata["filing_type"] = doc["filing_type"]
                if "filing_date" in doc:
                    metadata["filing_date"] = doc["filing_date"]
                if "metadata" in doc and isinstance(doc["metadata"], dict):
                    # Convert complex metadata to string to avoid Pinecone limitations
                    for k, v in doc["metadata"].items():
                        if isinstance(v, (dict, list)):
                            metadata[k] = json.dumps(v)
                        else:
                            metadata[k] = str(v)
                
                # Add text snippet (truncated)
                content = doc.get("content", "")
                metadata["text_snippet"] = content[:1000] if content else ""
                
                # Create vector
                vector = {
                    "id": doc_id,
                    "values": embedding,
                    "metadata": metadata
                }
                
                vectors.append(vector)
            
            # Batch upsert (500 at a time to stay within limits)
            batch_size = 500
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"Upserted {len(vectors)} documents to Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Error upserting to Pinecone: {str(e)}")
            return False
    
    async def query(self, 
                    query_embedding: List[float], 
                    filter_dict: Optional[Dict[str, Any]] = None, 
                    top_k: int = 5,
                    include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar documents
        
        Args:
            query_embedding: Query embedding vector
            filter_dict: Optional filter criteria
            top_k: Number of results to return
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of matching documents
        """
        if not self.index:
            logger.error("Pinecone index not initialized")
            return []
        
        try:
            # Execute query
            query_response = self.index.query(
                vector=query_embedding,
                filter=filter_dict,
                top_k=top_k,
                include_metadata=include_metadata
            )
            
            # Process results
            results = []
            for match in query_response.matches:
                result = {
                    "id": match.id,
                    "score": match.score,
                }
                
                if include_metadata and hasattr(match, 'metadata'):
                    result["metadata"] = match.metadata
                    
                    # Parse any JSON strings in metadata
                    for k, v in result["metadata"].items():
                        if isinstance(v, str) and v.startswith('{') and v.endswith('}'):
                            try:
                                result["metadata"][k] = json.loads(v)
                            except:
                                pass
                
                results.append(result)
            
            logger.info(f"Query returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error querying Pinecone: {str(e)}")
            return []
    
    async def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete documents from the vector store
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            Success status
        """
        if not self.index:
            logger.error("Pinecone index not initialized")
            return False
        
        try:
            # Delete by IDs
            self.index.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting from Pinecone: {str(e)}")
            return False
    
    async def delete_by_filter(self, filter_dict: Dict[str, Any]) -> bool:
        """
        Delete documents matching a filter
        
        Args:
            filter_dict: Filter criteria
            
        Returns:
            Success status
        """
        if not self.index:
            logger.error("Pinecone index not initialized")
            return False
        
        try:
            # Delete by filter
            self.index.delete(filter=filter_dict)
            logger.info(f"Deleted documents matching filter: {filter_dict}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting from Pinecone: {str(e)}")
            return False