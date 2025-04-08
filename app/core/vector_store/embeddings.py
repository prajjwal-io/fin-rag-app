import logging
from typing import List, Dict, Any
import openai
from ...config.settings import settings

logger = logging.getLogger(__name__)

class OpenAIEmbeddings:
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.EMBEDDING_MODEL
        openai.api_key = self.api_key
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        try:
            # Check for empty texts
            valid_texts = [text for text in texts if text and isinstance(text, str)]
            if not valid_texts:
                logger.warning("No valid texts provided for embedding")
                return []
            
            # Create embeddings using OpenAI
            response = openai.embeddings.create(
                model=self.model,
                input=valid_texts
            )
            
            # Extract embeddings from response
            embeddings = [item.embedding for item in response.data]
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return []
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text string
            
        Returns:
            Embedding vector
        """
        embeddings = await self.get_embeddings([text])
        return embeddings[0] if embeddings else []
    
    async def embed_documents(self, documents: List[Dict[str, Any]]) -> tuple:
        """
        Extract content and generate embeddings for a list of documents
        
        Args:
            documents: List of document dictionaries with 'content' field
            
        Returns:
            Tuple of (documents, embeddings)
        """
        try:
            # Extract text content from documents
            texts = [doc.get('content', '') for doc in documents]
            
            # Generate embeddings
            embeddings = await self.get_embeddings(texts)
            
            return documents, embeddings
            
        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}")
            return documents, []