from typing import List, Dict, Any
import re
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ...config.settings import settings

logger = logging.getLogger(__name__)

class TextChunker:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    async def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a document into manageable chunks for embedding
        
        Args:
            document: Document to chunk with 'content' field
            
        Returns:
            List of document chunks with metadata
        """
        try:
            content = document.get('content', '')
            if not content:
                logger.warning(f"Empty content for document: {document.get('source', 'unknown')}")
                return []
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(content)
            
            # Create document chunks with metadata
            document_chunks = []
            for i, chunk_text in enumerate(chunks):
                # Create a copy of the original document
                chunk_doc = document.copy()
                
                # Replace content with chunk
                chunk_doc['content'] = chunk_text
                
                # Add chunk metadata
                chunk_doc['chunk_id'] = i
                chunk_doc['chunk_count'] = len(chunks)
                
                document_chunks.append(chunk_doc)
            
            logger.info(f"Split document into {len(chunks)} chunks")
            return document_chunks
            
        except Exception as e:
            logger.error(f"Error chunking document: {str(e)}")
            return []
    
    async def chunk_financial_data(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process and chunk a list of financial documents
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        chunked_docs = []
        
        for doc in documents:
            chunks = await self.chunk_document(doc)
            chunked_docs.extend(chunks)
        
        return chunked_docs