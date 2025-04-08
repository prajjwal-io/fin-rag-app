import pytest
import asyncio
from typing import Dict, Any
from app.core.document_processing.text_chunker import TextChunker
from app.core.document_processing.metadata_extractor import MetadataExtractor

pytestmark = pytest.mark.asyncio

class TestTextChunker:
    async def test_chunk_document(self, text_chunker: TextChunker, sample_document: Dict[str, Any]):
        """
        Test chunking a document.
        """
        # Chunk the sample document
        chunked_docs = await text_chunker.chunk_document(sample_document)
        
        # Verify that we got some chunks
        assert len(chunked_docs) > 0
        
        # Verify the structure of the first chunk
        first_chunk = chunked_docs[0]
        assert "ticker" in first_chunk
        assert "content" in first_chunk
        assert "chunk_id" in first_chunk
        assert "chunk_count" in first_chunk
        
        # Verify chunk content is a subset of the original content
        assert len(first_chunk["content"]) <= len(sample_document["content"])
        assert first_chunk["ticker"] == sample_document["ticker"]

class TestMetadataExtractor:
    def test_extract_financial_periods(self, metadata_extractor: MetadataExtractor):
        """
        Test extracting financial periods from text.
        """
        # Test text with various financial periods
        text = """
        For the fiscal year ended December 31, 2023.
        Results for Q4 2023 were strong.
        The first quarter of 2023 showed improvement over Q4 2022.
        """
        
        periods = metadata_extractor.extract_financial_periods(text)
        
        # Verify that we extracted some periods
        assert len(periods) > 0
        
        # Verify the types of periods extracted
        period_types = [p[0] for p in periods]
        assert "fiscal_year" in period_types or "quarter" in period_types or "date" in period_types
    
    def test_extract_financial_entities(self, metadata_extractor: MetadataExtractor):
        """
        Test extracting financial entities from text.
        """
        # Skip if spaCy model is not available
        if not hasattr(metadata_extractor, 'nlp') or metadata_extractor.nlp is None:
            pytest.skip("spaCy model not available")
        
        # Test text with financial entities
        text = """
        Apple Inc. reported strong quarterly earnings, with revenue reaching $89.5 billion.
        The company's gross margin was 45.2%, and operating margin was 30.1%.
        EPS increased to $1.46, up from $1.29 in the same quarter last year.
        """
        
        entities = metadata_extractor.extract_financial_entities(text)
        
        # Verify that we extracted some entities
        assert len(entities["companies"]) > 0 or len(entities["organizations"]) > 0 or len(entities["financial_terms"]) > 0

