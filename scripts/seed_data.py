
import asyncio
import logging
import sys
import os
import json
from datetime import datetime, timedelta
import random

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config.settings import settings
from app.services.research_service import ResearchService
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
from app.db.mongodb import get_database, store_document_metadata, upsert_company

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# List of sample companies to seed data for
SAMPLE_COMPANIES = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

async def seed_data():
    """
    Seed sample data for testing
    """
    # Get database connection
    db = await get_database()
    
    # Initialize services
    sec_edgar_ingestion = SECEdgarIngestion()
    market_data_ingestion = MarketDataIngestion()
    text_chunker = TextChunker()
    metadata_extractor = MetadataExtractor()
    openai_embeddings = OpenAIEmbeddings()
    pinecone_vector_store = PineconeVectorStore()
    document_retriever = DocumentRetriever()
    rag_query_engine = RAGQueryEngine()
    query_augmentation = QueryAugmentation()
    sentiment_analyzer = SentimentAnalyzer()
    entity_extractor = EntityExtractor()
    financial_metrics_analyzer = FinancialMetricsAnalyzer()
    
    # Create research service
    research_service = ResearchService(
        db=db,
        sec_edgar_ingestion=sec_edgar_ingestion,
        market_data_ingestion=market_data_ingestion,
        text_chunker=text_chunker,
        metadata_extractor=metadata_extractor,
        openai_embeddings=openai_embeddings,
        pinecone_vector_store=pinecone_vector_store,
        document_retriever=document_retriever,
        rag_query_engine=rag_query_engine,
        query_augmentation=query_augmentation,
        sentiment_analyzer=sentiment_analyzer,
        entity_extractor=entity_extractor,
        financial_metrics_analyzer=financial_metrics_analyzer
    )
    
    # Process each sample company
    for ticker in SAMPLE_COMPANIES:
        logger.info(f"Seeding data for {ticker}")
        
        try:
            # Ingest company data
            await research_service.ingest_company_data(
                ticker=ticker,
                filing_types=["10-K", "10-Q"],  # Limited for seeding
                limit_per_type=1,  # Limited for seeding
                include_news=True,
                include_financials=True
            )
            
            logger.info(f"Data ingestion completed for {ticker}")
            
        except Exception as e:
            logger.error(f"Error seeding data for {ticker}: {str(e)}")
    
    logger.info("Data seeding completed")

if __name__ == "__main__":
    logger.info("Starting data seeding")
    asyncio.run(seed_data())
    logger.info("Data seeding completed")