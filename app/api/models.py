from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

# ==================== Query Models ====================

class QueryRequest(BaseModel):
    """
    Model for a query request
    """
    query: str = Field(..., description="The query or question to ask")
    ticker: Optional[str] = Field(None, description="Optional ticker symbol to focus on")
    content_types: Optional[List[str]] = Field(None, description="Optional list of content types to search (sec_filing, news, financial_data)")
    expand_query: bool = Field(True, description="Whether to expand the query with financial terms")

class QuerySource(BaseModel):
    """
    Model for a source reference in a query response
    """
    type: str = Field(..., description="Source type (sec_filing, news, financial_data)")
    source: str = Field(..., description="Source identifier")
    filing_type: Optional[str] = Field(None, description="Filing type (10-K, 10-Q, etc.) if available")
    filing_date: Optional[str] = Field(None, description="Filing date if available")

class QueryResponse(BaseModel):
    """
    Model for a query response
    """
    answer: str = Field(..., description="Answer to the query")
    sources: List[QuerySource] = Field([], description="List of sources used for the answer")
    query: str = Field(..., description="The original query")
    expanded_query: Optional[str] = Field(None, description="The expanded query if query expansion was used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the response")

# ==================== Company Research Models ====================

class CompanyResearchRequest(BaseModel):
    """
    Model for a company research request
    """
    ticker: str = Field(..., description="Company ticker symbol")
    topics: List[str] = Field([], description="List of topics to research")
    time_period: Optional[str] = Field(None, description="Time period to focus on (e.g., '2022', 'last 2 years')")

class CompanyResearchResponse(BaseModel):
    """
    Model for a company research response
    """
    status: str = Field(..., description="Status of the research request (processing, completed, error)")
    message: str = Field(..., description="Status message")
    report_id: str = Field(..., description="Unique ID for the research report")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the response")

class ResearchReport(BaseModel):
    """
    Model for a research report
    """
    report_id: str = Field(..., description="Unique ID for the research report")
    ticker: str = Field(..., description="Company ticker symbol")
    topics: List[str] = Field(..., description="List of topics covered in the report")
    time_period: Optional[str] = Field(None, description="Time period covered in the report")
    sections: List[Dict[str, Any]] = Field(..., description="Sections of the report")
    summary: str = Field(..., description="Executive summary of the report")
    sources: List[Dict[str, Any]] = Field(..., description="Sources used in the report")
    status: str = Field(..., description="Status of the report (processing, completed, error)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the report")

# ==================== Document Models ====================

class IngestDocumentRequest(BaseModel):
    """
    Model for a document ingestion request
    """
    document: Dict[str, Any] = Field(..., description="Document to ingest")

class DocumentIndexStatus(BaseModel):
    """
    Model for document indexing status
    """
    status: str = Field(..., description="Status of the document indexing (processing, completed, error)")
    message: str = Field(..., description="Status message")
    document_id: Optional[str] = Field(None, description="Document ID if available")
    ticker: Optional[str] = Field(None, description="Ticker symbol if available")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the status")

# ==================== Financial Analysis Models ====================

class FinancialMetricsRequest(BaseModel):
    """
    Model for a financial metrics analysis request
    """
    ticker: str = Field(..., description="Company ticker symbol")
    metric_type: str = Field(..., description="Type of metric to analyze (revenue, profit, margins, growth)")
    time_period: Optional[str] = Field(None, description="Time period to analyze")

class FinancialMetric(BaseModel):
    """
    Model for a financial metric
    """
    name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    unit: str = Field(..., description="Metric unit (%, $, etc.)")
    time_period: Optional[str] = Field(None, description="Time period the metric applies to")
    comparison: Optional[Dict[str, Any]] = Field(None, description="Comparison to previous period or industry average")

class FinancialAnalysisResponse(BaseModel):
    """
    Model for a financial analysis response
    """
    ticker: str = Field(..., description="Company ticker symbol")
    metrics: List[FinancialMetric] = Field(..., description="List of analyzed metrics")
    analysis: str = Field(..., description="Textual analysis of the metrics")
    sources: List[QuerySource] = Field([], description="Sources used for the analysis")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the analysis")