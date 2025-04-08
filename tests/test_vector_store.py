import pytest
import asyncio
from typing import Dict, Any, List
from app.core.vector_store.pinecone_client import PineconeVectorStore
from app.core.vector_store.embeddings import OpenAIEmbeddings

pytestmark = pytest.mark.asyncio

class TestOpenAIEmbeddings:
    async def test_get_embedding(self, openai_embeddings: OpenAIEmbeddings):
        """
        Test generating an embedding for a text.
        """
        # Skip if no API key is provided
        if not openai_embeddings.api_key:
            pytest.skip("OpenAI API key not provided")
        
        # Generate embedding for a simple text
        text = "Apple reported strong quarterly earnings."
        embedding = await openai_embeddings.get_embedding(text)
        
        # Verify that we got an embedding
        assert len(embedding) > 0
        
        # Verify that the embedding has the correct dimension
        assert len(embedding) == openai_embeddings.model_dimension

class TestPineconeVectorStore:
    async def test_init_connection(self, pinecone_vector_store: PineconeVectorStore):
        """
        Test initializing connection to Pinecone.
        """
        # Skip if no API key is provided
        if not pinecone_vector_store.api_key:
            pytest.skip("Pinecone API key not provided")
        
        # Verify that the connection was initialized
        assert pinecone_vector_store.pc is not None
        
        # Verify that the index exists or was created
        assert pinecone_vector_store.index is not None
    
    async def test_upsert_and_query(self, pinecone_vector_store: PineconeVectorStore, openai_embeddings: OpenAIEmbeddings):
        """
        Test upserting documents and querying the vector store.
        """
        # Skip if no API keys are provided
        if not pinecone_vector_store.api_key or not openai_embeddings.api_key:
            pytest.skip("Pinecone or OpenAI API key not provided")
        
        # Create test documents
        documents = [
            {
                "ticker": "TEST",
                "content": "This is a test document for Pinecone vector store.",
                "content_type": "test",
                "source": "test_source"
            }
        ]
        
        # Generate embeddings
        texts = [doc["content"] for doc in documents]
        embeddings = await openai_embeddings.get_embeddings(texts)
        
        # Upsert documents
        success = await pinecone_vector_store.upsert_documents(documents, embeddings)
        assert success
        
        # Query the vector store
        query_embedding = await openai_embeddings.get_embedding("test document")
        results = await pinecone_vector_store.query(query_embedding, top_k=1)
        
        # Verify that we got some results
        assert len(results) > 0
        
        # Clean up
        ids = [result["id"] for result in results]
        await pinecone_vector_store.delete_documents(ids)

