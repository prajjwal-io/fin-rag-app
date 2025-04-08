import pytest
import asyncio
from typing import Dict, Any, AsyncGenerator
import os
import sys
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config.settings import settings
from app.core.data_ingestion.sec_edgar import SECEdgarIngestion
from app.core.data_ingestion.market_data import MarketDataIngestion
from app.core.document_processing.text_chunker import TextChunker
from app.core.document_processing.metadata_extractor import MetadataExtractor
from app.core.vector_store.embeddings import OpenAIEmbeddings
from app.core.vector_store.pinecone_client import PineconeVectorStore
from app.core.rag.retriever import DocumentRetriever
from app.core.rag.query_engine import RAGQueryEngine
from app.core.rag.augmentation import QueryAugmentation
from app.core.financial_nlp.sentiment_analyzer import SentimentAnalyzer
from app.core.financial_nlp.entity_extractor import EntityExtractor
from app.core.financial_nlp.financial_metrics import FinancialMetricsAnalyzer

# Test database name - use a different database for testing
TEST_DB_NAME = "financial_research_test_db"

@pytest.fixture(scope="session")
def event_loop():
    """
    Create an instance of the default event loop for each test case.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def db() -> AsyncGenerator[AsyncIOMotorDatabase, None]:
    """
    Fixture that provides a test database.
    """
    # Use the same MongoDB connection but a different database for testing
    client = AsyncIOMotorClient(settings.MONGODB_URI)
    database = client[TEST_DB_NAME]
    
    # Clear database before tests
    await client.drop_database(TEST_DB_NAME)
    
    yield database
    
    # Clean up after tests
    await client.drop_database(TEST_DB_NAME)
    client.close()

@pytest.fixture
def sample_document() -> Dict[str, Any]:
    """
    Fixture that provides a sample document.
    """
    return {
        "ticker": "AAPL",
        "content_type": "sec_filing",
        "filing_type": "10-K",
        "filing_date": "2023-10-30",
        "source": "https://www.sec.gov/Archives/edgar/data/0000320193/000032019323000106/aapl-20230930.htm",
        "content": """
        Apple Inc.
        
        FORM 10-K
        
        For the fiscal year ended September 30, 2023
        
        PART I
        
        Item 1. Business
        
        Apple Inc. designs, manufactures and markets smartphones, personal computers, tablets, wearables and accessories, and sells a variety of related services. The Company's products include iPhone, Mac, iPad, and Wearables, Home and Accessories. The Company operates various platforms that allow third parties to sell digital content and services within the Company's products. The Company's reportable segments are the Americas, Europe, Greater China, Japan and Rest of Asia Pacific.
        """
    }

@pytest.fixture
def sample_news_article() -> Dict[str, Any]:
    """
    Fixture that provides a sample news article.
    """
    return {
        "ticker": "AAPL",
        "headline": "Apple Reports Record Fourth Quarter Revenue",
        "summary": "Apple today announced financial results for its fiscal 2023 fourth quarter ended September 30, 2023. The Company posted quarterly revenue of $89.5 billion, up 8 percent year over year, and quarterly earnings per diluted share of $1.46, up 13 percent year over year.",
        "url": "https://www.apple.com/newsroom/2023/11/apple-reports-fourth-quarter-results/",
        "datetime": "2023-11-02 16:30:00",
        "source": "Apple Newsroom",
        "content_type": "news"
    }

@pytest.fixture
def sec_edgar_ingestion() -> SECEdgarIngestion:
    """
    Fixture that provides a SECEdgarIngestion instance.
    """
    return SECEdgarIngestion()

@pytest.fixture
def market_data_ingestion() -> MarketDataIngestion:
    """
    Fixture that provides a MarketDataIngestion instance.
    """
    return MarketDataIngestion()

@pytest.fixture
def text_chunker() -> TextChunker:
    """
    Fixture that provides a TextChunker instance.
    """
    return TextChunker()

@pytest.fixture
def metadata_extractor() -> MetadataExtractor:
    """
    Fixture that provides a MetadataExtractor instance.
    """
    return MetadataExtractor()

@pytest.fixture
def openai_embeddings() -> OpenAIEmbeddings:
    """
    Fixture that provides an OpenAIEmbeddings instance.
    """
    return OpenAIEmbeddings()

@pytest.fixture
def pinecone_vector_store() -> PineconeVectorStore:
    """
    Fixture that provides a PineconeVectorStore instance.
    """
    return PineconeVectorStore()

@pytest.fixture
def document_retriever() -> DocumentRetriever:
    """
    Fixture that provides a DocumentRetriever instance.
    """
    return DocumentRetriever()

@pytest.fixture
def rag_query_engine() -> RAGQueryEngine:
    """
    Fixture that provides a RAGQueryEngine instance.
    """
    return RAGQueryEngine()

@pytest.fixture
def query_augmentation() -> QueryAugmentation:
    """
    Fixture that provides a QueryAugmentation instance.
    """
    return QueryAugmentation()

@pytest.fixture
def sentiment_analyzer() -> SentimentAnalyzer:
    """
    Fixture that provides a SentimentAnalyzer instance.
    """
    return SentimentAnalyzer()

@pytest.fixture
def entity_extractor() -> EntityExtractor:
    """
    Fixture that provides an EntityExtractor instance.
    """
    return EntityExtractor()

@pytest.fixture
def financial_metrics_analyzer() -> FinancialMetricsAnalyzer:
    """
    Fixture that provides a FinancialMetricsAnalyzer instance.
    """
    return FinancialMetricsAnalyzer()