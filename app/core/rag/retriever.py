from typing import List, Dict, Any, Optional
import logging
from ...config.settings import settings
from ..vector_store.pinecone_client import PineconeVectorStore
from ..vector_store.embeddings import OpenAIEmbeddings

logger = logging.getLogger(__name__)

class DocumentRetriever:
    def __init__(self):
        self.vector_store = PineconeVectorStore()
        self.embeddings = OpenAIEmbeddings()
        self.max_documents = settings.MAX_DOCUMENTS_RETRIEVED
    
    async def retrieve_documents(self, 
                                query: str, 
                                filters: Optional[Dict[str, Any]] = None, 
                                top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Query string
            filters: Optional metadata filters
            top_k: Optional number of results to return
            
        Returns:
            List of retrieved documents
        """
        try:
            # Set default top_k if not provided
            if top_k is None:
                top_k = self.max_documents
            
            # Generate embedding for query
            query_embedding = await self.embeddings.get_embedding(query)
            
            if not query_embedding:
                logger.error("Failed to generate embedding for query")
                return []
            
            # Query vector store
            results = await self.vector_store.query(
                query_embedding=query_embedding,
                filter_dict=filters,
                top_k=top_k,
                include_metadata=True
            )
            
            # Process results
            documents = []
            for result in results:
                if 'metadata' in result:
                    # Extract text content from metadata
                    text = result['metadata'].get('text_snippet', '')
                    
                    document = {
                        'id': result['id'],
                        'content': text,
                        'score': result['score'],
                        'metadata': result['metadata']
                    }
                    
                    documents.append(document)
            
            logger.info(f"Retrieved {len(documents)} documents for query")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    async def retrieve_by_ticker(self, 
                                query: str, 
                                ticker: str, 
                                content_types: Optional[List[str]] = None,
                                top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve documents for a specific ticker symbol
        
        Args:
            query: Query string
            ticker: Stock ticker symbol
            content_types: Optional list of content types to filter by
            top_k: Optional number of results to return
            
        Returns:
            List of retrieved documents
        """
        # Build filter
        filters = {"ticker": ticker}
        
        if content_types:
            filters["content_type"] = {"$in": content_types}
        
        return await self.retrieve_documents(query, filters, top_k)
    
    async def retrieve_financial_documents(self,
                                         query: str,
                                         ticker: Optional[str] = None,
                                         filing_type: Optional[str] = None,
                                         date_range: Optional[tuple] = None,
                                         top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve financial documents with specific criteria
        
        Args:
            query: Query string
            ticker: Optional ticker symbol
            filing_type: Optional filing type (10-K, 10-Q, etc.)
            date_range: Optional tuple of (start_date, end_date)
            top_k: Optional number of results to return
            
        Returns:
            List of retrieved documents
        """
        # Build filters
        filters = {}
        
        if ticker:
            filters["ticker"] = ticker
        
        if filing_type:
            filters["filing_type"] = filing_type
        
        # Date filtering would need special handling since it's a string in metadata
        # This would require a more complex filter strategy
        
        return await self.retrieve_documents(query, filters, top_k)