import pytest
import asyncio
from typing import Dict, Any
from app.core.rag.retriever import DocumentRetriever
from app.core.rag.query_engine import RAGQueryEngine
from app.core.rag.augmentation import QueryAugmentation

pytestmark = pytest.mark.asyncio

class TestDocumentRetriever:
    async def test_retrieve_documents(self, document_retriever: DocumentRetriever):
        """
        Test retrieving documents.
        """
        # Skip if no API keys are provided
        if not document_retriever.vector_store.api_key or not document_retriever.embeddings.api_key:
            pytest.skip("API keys not provided")
        
        # This is more of an integration test, so we'll keep it simple
        pytest.skip("Integration test that requires populated vector store")

class TestRAGQueryEngine:
    async def test_format_retrieved_documents(self, rag_query_engine: RAGQueryEngine):
        """
        Test formatting retrieved documents.
        """
        # Create test documents
        documents = [
            {
                "content": "This is the first test document.",
                "metadata": {
                    "source": "test_source_1",
                    "content_type": "sec_filing",
                    "filing_type": "10-K",
                    "filing_date": "2023-10-30"
                }
            },
            {
                "content": "This is the second test document.",
                "metadata": {
                    "source": "test_source_2",
                    "content_type": "news",
                    "filing_date": "2023-11-01"
                }
            }
        ]
        
        # Format documents
        formatted_context = rag_query_engine.format_retrieved_documents(documents)
        
        # Verify that we got a formatted context
        assert len(formatted_context) > 0
        
        # Verify that the context contains the document contents
        assert "first test document" in formatted_context
        assert "second test document" in formatted_context

class TestQueryAugmentation:
    async def test_expand_financial_query(self, query_augmentation: QueryAugmentation):
        """
        Test expanding a financial query.
        """
        # Skip if no API key is provided
        if not getattr(query_augmentation.llm, 'api_key', None):
            pytest.skip("OpenAI API key not provided")
        
        # Expand a simple query
        query = "What was Apple's revenue last quarter?"
        expanded_query = await query_augmentation.expand_financial_query(query)
        
        # Verify that we got an expanded query
        assert len(expanded_query) > 0
        
        # Verify that the expanded query contains more financial terms
        assert len(expanded_query) > len(query)
