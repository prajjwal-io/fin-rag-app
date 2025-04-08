from fastapi import Depends
from typing import Generator, Optional
from ..services.research_service import ResearchService
from ..core.data_ingestion.sec_edgar import SECEdgarIngestion
from ..core.data_ingestion.market_data import MarketDataIngestion
from ..core.document_processing.text_chunker import TextChunker
from ..core.document_processing.metadata_extractor import MetadataExtractor
from ..core.vector_store.embeddings import OpenAIEmbeddings
from ..core.vector_store.pinecone_client import PineconeVectorStore
from ..core.rag.retriever import DocumentRetriever
from ..core.rag.query_engine import RAGQueryEngine
from ..core.rag.augmentation import QueryAugmentation
from ..core.financial_nlp.sentiment_analyzer import SentimentAnalyzer
from ..core.financial_nlp.entity_extractor import EntityExtractor
from ..core.financial_nlp.financial_metrics import FinancialMetricsAnalyzer
from ..db.mongodb import get_database

# Dependency for SEC Edgar data ingestion
def get_sec_edgar_ingestion() -> SECEdgarIngestion:
    return SECEdgarIngestion()

# Dependency for market data ingestion
def get_market_data_ingestion() -> MarketDataIngestion:
    return MarketDataIngestion()

# Dependency for text chunker
def get_text_chunker() -> TextChunker:
    return TextChunker()

# Dependency for metadata extractor
def get_metadata_extractor() -> MetadataExtractor:
    return MetadataExtractor()

# Dependency for OpenAI embeddings
def get_openai_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings()

# Dependency for Pinecone vector store
def get_pinecone_vector_store() -> PineconeVectorStore:
    return PineconeVectorStore()

# Dependency for document retriever
def get_document_retriever() -> DocumentRetriever:
    return DocumentRetriever()

# Dependency for RAG query engine
def get_rag_query_engine() -> RAGQueryEngine:
    return RAGQueryEngine()

# Dependency for query augmentation
def get_query_augmentation() -> QueryAugmentation:
    return QueryAugmentation()

# Dependency for sentiment analyzer
def get_sentiment_analyzer() -> SentimentAnalyzer:
    return SentimentAnalyzer()

# Dependency for entity extractor
def get_entity_extractor() -> EntityExtractor:
    return EntityExtractor()

# Dependency for financial metrics analyzer
def get_financial_metrics_analyzer() -> FinancialMetricsAnalyzer:
    return FinancialMetricsAnalyzer()

# Dependency for research service
def get_research_service(
    db=Depends(get_database),
    sec_edgar_ingestion=Depends(get_sec_edgar_ingestion),
    market_data_ingestion=Depends(get_market_data_ingestion),
    text_chunker=Depends(get_text_chunker),
    metadata_extractor=Depends(get_metadata_extractor),
    openai_embeddings=Depends(get_openai_embeddings),
    pinecone_vector_store=Depends(get_pinecone_vector_store),
    document_retriever=Depends(get_document_retriever),
    rag_query_engine=Depends(get_rag_query_engine),
    query_augmentation=Depends(get_query_augmentation),
    sentiment_analyzer=Depends(get_sentiment_analyzer),
    entity_extractor=Depends(get_entity_extractor),
    financial_metrics_analyzer=Depends(get_financial_metrics_analyzer)
) -> ResearchService:
    return ResearchService(
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