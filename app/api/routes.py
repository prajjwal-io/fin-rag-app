from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import logging
import json
from datetime import datetime

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
from ..services.research_service import ResearchService
from .models import (
    QueryRequest,
    QueryResponse,
    CompanyResearchRequest,
    CompanyResearchResponse,
    IngestDocumentRequest,
    DocumentIndexStatus,
    FinancialMetricsRequest,
    FinancialAnalysisResponse
)
from .dependencies import get_research_service

router = APIRouter()
logger = logging.getLogger(__name__)

# ==================== Company Data Endpoints ====================

@router.post("/companies/{ticker}/ingest", response_model=DocumentIndexStatus)
async def ingest_company_data(
    ticker: str, 
    background_tasks: BackgroundTasks,
    filing_types: List[str] = Query(["10-K", "10-Q"]),
    limit_per_type: int = Query(3),
    include_news: bool = Query(True),
    include_financials: bool = Query(True),
    research_service: ResearchService = Depends(get_research_service)
):
    """
    Ingest company data from SEC filings, news, and financial statements
    
    Args:
        ticker: Company ticker symbol
        filing_types: List of filing types to ingest
        limit_per_type: Limit of filings per type
        include_news: Whether to include news
        include_financials: Whether to include financial data
    
    Returns:
        Status of the ingestion process
    """
    # Start the ingestion process in the background
    background_tasks.add_task(
        research_service.ingest_company_data,
        ticker=ticker,
        filing_types=filing_types,
        limit_per_type=limit_per_type,
        include_news=include_news,
        include_financials=include_financials
    )
    
    return {
        "status": "processing",
        "message": f"Started ingestion process for {ticker}. This may take a few minutes.",
        "ticker": ticker,
        "timestamp": datetime.now().isoformat()
    }

@router.get("/companies/{ticker}/data", response_model=Dict[str, Any])
async def get_company_data(
    ticker: str,
    research_service: ResearchService = Depends(get_research_service)
):
    """
    Get available data for a company
    
    Args:
        ticker: Company ticker symbol
    
    Returns:
        Summary of available data for the company
    """
    data = await research_service.get_company_data_summary(ticker)
    if not data:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
    
    return data

# ==================== Document Endpoints ====================

@router.post("/documents/ingest", response_model=DocumentIndexStatus)
async def ingest_document(
    request: IngestDocumentRequest,
    background_tasks: BackgroundTasks,
    research_service: ResearchService = Depends(get_research_service)
):
    """
    Ingest a document into the system
    
    Args:
        request: Document ingestion request
    
    Returns:
        Status of the ingestion process
    """
    # Start the ingestion process in the background
    background_tasks.add_task(
        research_service.ingest_document,
        document=request.document
    )
    
    return {
        "status": "processing",
        "message": "Started document ingestion process. This may take a few minutes.",
        "document_id": request.document.get("id", "unknown"),
        "ticker": request.document.get("ticker", "unknown"),
        "timestamp": datetime.now().isoformat()
    }

@router.post("/documents/upload", response_model=DocumentIndexStatus)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    ticker: Optional[str] = None,
    content_type: Optional[str] = "document",
    research_service: ResearchService = Depends(get_research_service)
):
    """
    Upload and ingest a document file
    
    Args:
        file: Document file to upload
        ticker: Company ticker symbol (optional)
        content_type: Document content type
    
    Returns:
        Status of the ingestion process
    """
    # Read file content
    content = await file.read()
    
    # Process different file types
    if file.filename.endswith(".pdf"):
        # Process PDF
        document = {
            "filename": file.filename,
            "content_type": content_type,
            "file_content": content,
            "file_type": "pdf"
        }
    elif file.filename.endswith(".txt"):
        # Process text file
        document = {
            "filename": file.filename,
            "content_type": content_type,
            "content": content.decode("utf-8"),
            "file_type": "text"
        }
    elif file.filename.endswith((".doc", ".docx")):
        # Process Word document
        document = {
            "filename": file.filename,
            "content_type": content_type,
            "file_content": content,
            "file_type": "docx"
        }
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    # Add ticker if provided
    if ticker:
        document["ticker"] = ticker
    
    # Start the ingestion process in the background
    background_tasks.add_task(
        research_service.ingest_uploaded_document,
        document=document
    )
    
    return {
        "status": "processing",
        "message": f"Started processing of {file.filename}. This may take a few minutes.",
        "document_id": file.filename,
        "ticker": ticker,
        "timestamp": datetime.now().isoformat()
    }

# ==================== Query Endpoints ====================

@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    research_service: ResearchService = Depends(get_research_service)
):
    """
    Query the system with a natural language question
    
    Args:
        request: Query request with question and optional parameters
    
    Returns:
        Answer to the question with source references
    """
    try:
        # Process the query
        response = await research_service.process_query(
            query=request.query,
            ticker=request.ticker,
            content_types=request.content_types,
            expand_query=request.expand_query
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@router.post("/research", response_model=CompanyResearchResponse)
async def research_company(
    request: CompanyResearchRequest,
    background_tasks: BackgroundTasks,
    research_service: ResearchService = Depends(get_research_service)
):
    """
    Generate a company research report
    
    Args:
        request: Company research request
    
    Returns:
        Research report
    """
    try:
        # Start the research process
        report_id = await research_service.start_company_research(
            ticker=request.ticker,
            topics=request.topics,
            time_period=request.time_period
        )
        
        # Generate the research report in the background
        background_tasks.add_task(
            research_service.generate_research_report,
            report_id=report_id
        )
        
        return {
            "status": "processing",
            "message": f"Started research report generation for {request.ticker}. This may take a few minutes.",
            "report_id": report_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting company research: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting company research: {str(e)}")

@router.get("/research/{report_id}", response_model=Dict[str, Any])
async def get_research_report(
    report_id: str,
    research_service: ResearchService = Depends(get_research_service)
):
    """
    Get a generated research report
    
    Args:
        report_id: Research report ID
    
    Returns:
        Research report
    """
    report = await research_service.get_research_report(report_id)
    
    if not report:
        raise HTTPException(status_code=404, detail=f"Research report {report_id} not found")
    
    return report

# ==================== Financial Analysis Endpoints ====================

@router.post("/financial-metrics", response_model=FinancialAnalysisResponse)
async def analyze_financial_metrics(
    request: FinancialMetricsRequest,
    research_service: ResearchService = Depends(get_research_service)
):
    """
    Analyze financial metrics for a company
    
    Args:
        request: Financial metrics analysis request
    
    Returns:
        Financial metrics analysis
    """
    try:
        # Process the financial metrics analysis
        response = await research_service.analyze_financial_metrics(
            ticker=request.ticker,
            metric_type=request.metric_type,
            time_period=request.time_period
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing financial metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing financial metrics: {str(e)}")

@router.get("/sentiment/{ticker}", response_model=Dict[str, Any])
async def analyze_sentiment(
    ticker: str,
    days: int = Query(30),
    research_service: ResearchService = Depends(get_research_service)
):
    """
    Analyze sentiment for a company
    
    Args:
        ticker: Company ticker symbol
        days: Number of days to analyze
    
    Returns:
        Sentiment analysis
    """
    try:
        # Process the sentiment analysis
        response = await research_service.analyze_company_sentiment(ticker, days)
        
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")