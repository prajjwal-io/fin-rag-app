from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class Company(BaseModel):
    """
    Model for company data
    """
    ticker: str
    name: Optional[str] = None
    latest_price: Optional[float] = None
    latest_price_date: Optional[str] = None
    document_count: int = 0
    last_updated: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class Document(BaseModel):
    """
    Model for document metadata
    """
    id: str = Field(..., alias="_id")
    ticker: Optional[str] = None
    content_type: str
    source: str
    filing_type: Optional[str] = None
    filing_date: Optional[str] = None
    headline: Optional[str] = None
    url: Optional[str] = None
    ingestion_date: str
    metadata: Optional[Dict[str, Any]] = None

class ResearchReport(BaseModel):
    """
    Model for research report
    """
    id: str = Field(..., alias="_id")
    report_id: str
    ticker: str
    topics: List[str]
    time_period: Optional[str] = None
    sections: List[Dict[str, Any]] = []
    summary: str = ""
    sources: List[Dict[str, Any]] = []
    status: str
    timestamp: str
    completed_timestamp: Optional[str] = None
    error_message: Optional[str] = None

class QueryRecord(BaseModel):
    """
    Model for query history
    """
    id: str = Field(..., alias="_id")
    query: str
    ticker: Optional[str] = None
    expanded_query: Optional[str] = None
    timestamp: str
    response_status: Optional[str] = None
    execution_time_ms: Optional[int] = None

class IngestionRecord(BaseModel):
    """
    Model for ingestion history
    """
    id: str = Field(..., alias="_id")
    ticker: str
    filing_types: List[str]
    limit_per_type: int
    include_news: bool
    include_financials: bool
    status: str
    timestamp: str
    document_count: int = 0
    message: Optional[str] = None