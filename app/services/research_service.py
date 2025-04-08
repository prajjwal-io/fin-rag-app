import logging
import uuid
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json
import asyncio
import tempfile
import os
from motor.motor_asyncio import AsyncIOMotorDatabase

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
from ..db.mongodb import (
    store_document_metadata,
    upsert_company,
    store_research_report,
    update_research_report,
    get_research_report,
    store_query,
    store_ingestion_record,
    update_ingestion_status,
    get_company,
    list_company_documents
)

logger = logging.getLogger(__name__)

class ResearchService:
    def __init__(
        self,
        db: AsyncIOMotorDatabase,
        sec_edgar_ingestion: SECEdgarIngestion,
        market_data_ingestion: MarketDataIngestion,
        text_chunker: TextChunker,
        metadata_extractor: MetadataExtractor,
        openai_embeddings: OpenAIEmbeddings,
        pinecone_vector_store: PineconeVectorStore,
        document_retriever: DocumentRetriever,
        rag_query_engine: RAGQueryEngine,
        query_augmentation: QueryAugmentation,
        sentiment_analyzer: SentimentAnalyzer,
        entity_extractor: EntityExtractor,
        financial_metrics_analyzer: FinancialMetricsAnalyzer
    ):
        self.db = db
        self.sec_edgar_ingestion = sec_edgar_ingestion
        self.market_data_ingestion = market_data_ingestion
        self.text_chunker = text_chunker
        self.metadata_extractor = metadata_extractor
        self.openai_embeddings = openai_embeddings
        self.pinecone_vector_store = pinecone_vector_store
        self.document_retriever = document_retriever
        self.rag_query_engine = rag_query_engine
        self.query_augmentation = query_augmentation
        self.sentiment_analyzer = sentiment_analyzer
        self.entity_extractor = entity_extractor
        self.financial_metrics_analyzer = financial_metrics_analyzer
    
    async def ingest_company_data(self, 
                                ticker: str, 
                                filing_types: List[str] = ["10-K", "10-Q"],
                                limit_per_type: int = 3,
                                include_news: bool = True,
                                include_financials: bool = True) -> str:
        """
        Ingest all data for a company
        
        Args:
            ticker: Company ticker symbol
            filing_types: List of filing types to ingest
            limit_per_type: Maximum number of filings per type
            include_news: Whether to include news
            include_financials: Whether to include financial data
            
        Returns:
            Ingestion record ID
        """
        try:
            # Create ingestion record
            ingestion_record = {
                "ticker": ticker,
                "filing_types": filing_types,
                "limit_per_type": limit_per_type,
                "include_news": include_news,
                "include_financials": include_financials,
                "status": "in_progress",
                "timestamp": datetime.now().isoformat(),
                "document_count": 0
            }
            
            ingestion_id = await store_ingestion_record(self.db, ingestion_record)
            
            # Track processed documents
            document_count = 0
            
            # Ingest SEC filings
            sec_filings = await self.sec_edgar_ingestion.get_filings_data(
                ticker=ticker,
                filing_types=filing_types,
                limit=limit_per_type
            )
            
            # Process and embed SEC filings
            if sec_filings:
                # Add metadata
                for filing in sec_filings:
                    # Extract metadata
                    filing = self.metadata_extractor.extract_metadata(filing)
                
                # Chunk documents
                chunked_filings = await self.text_chunker.chunk_financial_data(sec_filings)
                
                # Generate embeddings
                documents, embeddings = await self.openai_embeddings.embed_documents(chunked_filings)
                
                # Store in Pinecone
                if documents and embeddings:
                    success = await self.pinecone_vector_store.upsert_documents(documents, embeddings)
                    if success:
                        document_count += len(documents)
                
                # Store metadata in MongoDB
                for filing in sec_filings:
                    filing_metadata = {
                        "ticker": ticker,
                        "content_type": "sec_filing",
                        "filing_type": filing.get("filing_type"),
                        "filing_date": filing.get("filing_date"),
                        "source": filing.get("source"),
                        "ingestion_date": datetime.now().isoformat()
                    }
                    await store_document_metadata(self.db, filing_metadata)
            
            # Ingest news if requested
            if include_news:
                news_articles = await self.market_data_ingestion.get_company_news(ticker, days=30)
                
                # Process and embed news
                if news_articles:
                    # Add metadata
                    for article in news_articles:
                        # Extract metadata
                        article = self.metadata_extractor.extract_metadata(article)
                    
                    # Chunk documents
                    chunked_news = await self.text_chunker.chunk_financial_data(news_articles)
                    
                    # Generate embeddings
                    documents, embeddings = await self.openai_embeddings.embed_documents(chunked_news)
                    
                    # Store in Pinecone
                    if documents and embeddings:
                        success = await self.pinecone_vector_store.upsert_documents(documents, embeddings)
                        if success:
                            document_count += len(documents)
                    
                    # Store metadata in MongoDB
                    for article in news_articles:
                        article_metadata = {
                            "ticker": ticker,
                            "content_type": "news",
                            "headline": article.get("headline"),
                            "source": article.get("source"),
                            "datetime": article.get("datetime"),
                            "url": article.get("url"),
                            "ingestion_date": datetime.now().isoformat()
                        }
                        await store_document_metadata(self.db, article_metadata)
            
            # Ingest financial data if requested
            if include_financials:
                financial_documents = await self.market_data_ingestion.format_financial_data_for_embedding(ticker)
                
                # Process and embed financial data
                if financial_documents:
                    # Add metadata
                    for document in financial_documents:
                        # Extract metadata
                        document = self.metadata_extractor.extract_metadata(document)
                    
                    # Chunk documents
                    chunked_financials = await self.text_chunker.chunk_financial_data(financial_documents)
                    
                    # Generate embeddings
                    documents, embeddings = await self.openai_embeddings.embed_documents(chunked_financials)
                    
                    # Store in Pinecone
                    if documents and embeddings:
                        success = await self.pinecone_vector_store.upsert_documents(documents, embeddings)
                        if success:
                            document_count += len(documents)
                    
                    # Store metadata in MongoDB
                    for document in financial_documents:
                        document_metadata = {
                            "ticker": ticker,
                            "content_type": "financial_data",
                            "source": document.get("source"),
                            "filing_date": document.get("filing_date"),
                            "ingestion_date": datetime.now().isoformat()
                        }
                        await store_document_metadata(self.db, document_metadata)
            
            # Update company record
            company_data = {
                "ticker": ticker,
                "last_updated": datetime.now().isoformat(),
                "document_count": document_count
            }
            
            # Get price data for basic company info
            try:
                price_data = await self.market_data_ingestion.get_stock_price_data(ticker, period="1mo")
                if price_data and len(price_data) > 0:
                    latest_price = price_data[-1]
                    company_data["latest_price"] = latest_price.get("close")
                    company_data["latest_price_date"] = latest_price.get("date")
            except Exception as e:
                logger.warning(f"Error getting price data for {ticker}: {str(e)}")
            
            await upsert_company(self.db, company_data)
            
            # Update ingestion record
            await update_ingestion_status(
                self.db,
                ingestion_id,
                "completed",
                f"Successfully ingested {document_count} documents for {ticker}"
            )
            
            return ingestion_id
            
        except Exception as e:
            logger.error(f"Error ingesting company data for {ticker}: {str(e)}")
            
            # Update ingestion record with error
            if 'ingestion_id' in locals():
                await update_ingestion_status(
                    self.db,
                    ingestion_id,
                    "error",
                    f"Error ingesting data: {str(e)}"
                )
            
            return ""
    
    async def ingest_document(self, document: Dict[str, Any]) -> bool:
        """
        Ingest a document into the system
        
        Args:
            document: Document to ingest
            
        Returns:
            Success status
        """
        try:
            # Extract metadata if not already present
            if 'metadata' not in document:
                document = self.metadata_extractor.extract_metadata(document)
            
            # Chunk document
            chunked_documents = await self.text_chunker.chunk_document(document)
            
            # Generate embeddings
            documents, embeddings = await self.openai_embeddings.embed_documents(chunked_documents)
            
            # Store in Pinecone
            if documents and embeddings:
                success = await self.pinecone_vector_store.upsert_documents(documents, embeddings)
                
                # Store metadata in MongoDB
                if success:
                    document_metadata = {
                        "content_type": document.get("content_type", "document"),
                        "source": document.get("source", document.get("filename", "unknown")),
                        "ingestion_date": datetime.now().isoformat()
                    }
                    
                    # Add ticker if available
                    if "ticker" in document:
                        document_metadata["ticker"] = document["ticker"]
                    
                    await store_document_metadata(self.db, document_metadata)
                
                return success
            
            return False
            
        except Exception as e:
            logger.error(f"Error ingesting document: {str(e)}")
            return False
    
    async def ingest_uploaded_document(self, document: Dict[str, Any]) -> bool:
        """
        Process and ingest an uploaded document
        
        Args:
            document: Document data with file content
            
        Returns:
            Success status
        """
        try:
            # Process document based on file type
            file_type = document.get("file_type")
            
            if file_type == "pdf":
                # Extract text from PDF
                from pypdf import PdfReader
                import io
                
                file_content = document.get("file_content")
                with io.BytesIO(file_content) as f:
                    reader = PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                
                document["content"] = text
            
            elif file_type == "docx":
                # Extract text from Word document
                import docx2txt
                import io
                
                file_content = document.get("file_content")
                with io.BytesIO(file_content) as f:
                    text = docx2txt.process(f)
                
                document["content"] = text
            
            # Content should already be set for text files
            
            # Clean up document object
            if "file_content" in document:
                del document["file_content"]
            
            # Set source to filename if not already set
            if "source" not in document:
                document["source"] = document.get("filename", "uploaded_document")
            
            # Ingest document
            return await self.ingest_document(document)
            
        except Exception as e:
            logger.error(f"Error processing uploaded document: {str(e)}")
            return False
    
    async def process_query(self, 
                         query: str, 
                         ticker: Optional[str] = None,
                         content_types: Optional[List[str]] = None,
                         expand_query: bool = True) -> Dict[str, Any]:
        """
        Process a query
        
        Args:
            query: User query
            ticker: Optional ticker symbol to focus on
            content_types: Optional list of content types to search
            expand_query: Whether to expand the query with financial terms
            
        Returns:
            Query response
        """
        try:
            # Log the query
            query_record = {
                "query": query,
                "ticker": ticker,
                "timestamp": datetime.now().isoformat()
            }
            await store_query(self.db, query_record)
            
            # Expand the query if requested
            expanded_query = None
            if expand_query:
                expanded_query = await self.query_augmentation.expand_financial_query(query)
                query_to_use = expanded_query
            else:
                query_to_use = query
            
            # Get query response from RAG engine
            response = await self.rag_query_engine.answer_question(
                query=query_to_use,
                ticker=ticker,
                content_types=content_types
            )
            
            # Format response
            result = {
                "answer": response.get("answer", ""),
                "sources": response.get("sources", []),
                "query": query,
                "expanded_query": expanded_query,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "answer": f"I encountered an error processing your query: {str(e)}",
                "sources": [],
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
    
    async def analyze_financial_metrics(self, 
                                     ticker: str, 
                                     metric_type: str,
                                     time_period: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze financial metrics for a company
        
        Args:
            ticker: Company ticker symbol
            metric_type: Type of metric to analyze
            time_period: Optional time period to analyze
            
        Returns:
            Financial analysis
        """
        try:
            # Use RAG query engine to analyze metrics
            analysis = await self.rag_query_engine.analyze_financial_metrics(
                ticker=ticker,
                metric_type=metric_type,
                time_period=time_period
            )
            
            # Extract metrics from the answer
            metrics = []
            
            # Basic metrics extraction from answer text
            answer_text = analysis.get("answer", "")
            
            # Extract revenue metrics if relevant
            if metric_type.lower() == "revenue":
                # Simple pattern matching for revenue numbers
                import re
                revenue_patterns = [
                    r'(\$[\d\.]+\s*(?:billion|million|B|M))',
                    r'([\d\.]+\s*(?:billion|million|B|M)\s*dollars)',
                    r'revenue of (\$[\d\.]+\s*(?:billion|million|B|M))'
                ]
                
                for pattern in revenue_patterns:
                    matches = re.findall(pattern, answer_text, re.IGNORECASE)
                    for match in matches:
                        metrics.append({
                            "name": "Revenue",
                            "value": self._parse_financial_value(match),
                            "unit": "$",
                            "time_period": time_period
                        })
            
            # Extract profit metrics if relevant
            elif metric_type.lower() in ["profit", "earnings"]:
                profit_patterns = [
                    r'(\$[\d\.]+\s*(?:billion|million|B|M))\s*(?:in|of)?\s*(?:profit|earnings|net income)',
                    r'(?:profit|earnings|net income) of (\$[\d\.]+\s*(?:billion|million|B|M))',
                    r'EPS of (\$[\d\.]+)'
                ]
                
                for pattern in profit_patterns:
                    matches = re.findall(pattern, answer_text, re.IGNORECASE)
                    for match in matches:
                        metrics.append({
                            "name": "Profit",
                            "value": self._parse_financial_value(match),
                            "unit": "$",
                            "time_period": time_period
                        })
            
            # Extract growth metrics if relevant
            elif metric_type.lower() == "growth":
                growth_patterns = [
                    r'([\d\.]+%)\s*(?:growth|increase)',
                    r'grew by ([\d\.]+%)',
                    r'growth of ([\d\.]+%)'
                ]
                
                for pattern in growth_patterns:
                    matches = re.findall(pattern, answer_text, re.IGNORECASE)
                    for match in matches:
                        metrics.append({
                            "name": "Growth Rate",
                            "value": float(match.replace("%", "")),
                            "unit": "%",
                            "time_period": time_period
                        })
            
            # Extract margin metrics if relevant
            elif metric_type.lower() == "margins":
                margin_patterns = [
                    r'([\d\.]+%)\s*(?:gross|net|operating|profit)?\s*margin',
                    r'(?:gross|net|operating|profit) margin of ([\d\.]+%)'
                ]
                
                for pattern in margin_patterns:
                    matches = re.findall(pattern, answer_text, re.IGNORECASE)
                    for match in matches:
                        metrics.append({
                            "name": "Margin",
                            "value": float(match.replace("%", "")),
                            "unit": "%",
                            "time_period": time_period
                        })
            
            # If no metrics were found but we have an answer, create a generic metric
            if not metrics and answer_text:
                metrics.append({
                    "name": metric_type.capitalize(),
                    "value": 0,  # Placeholder value
                    "unit": "$" if metric_type.lower() in ["revenue", "profit", "earnings"] else "%",
                    "time_period": time_period
                })
            
            # Format response
            result = {
                "ticker": ticker,
                "metrics": metrics,
                "analysis": analysis.get("answer", ""),
                "sources": analysis.get("sources", []),
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing financial metrics: {str(e)}")
            return {
                "ticker": ticker,
                "metrics": [],
                "analysis": f"I encountered an error analyzing {metric_type} for {ticker}: {str(e)}",
                "sources": [],
                "timestamp": datetime.now().isoformat()
            }
    
    def _parse_financial_value(self, value_text: str) -> float:
        """
        Parse a financial value from text
        
        Args:
            value_text: Financial value text (e.g., "$1.2 billion")
            
        Returns:
            Numeric value
        """
        try:
            # Clean up text
            clean_text = value_text.replace("$", "").replace(",", "").strip()
            
            # Extract number and unit
            import re
            match = re.match(r"([\d\.]+)\s*(billion|million|B|M)?", clean_text, re.IGNORECASE)
            
            if match:
                number = float(match.group(1))
                unit = match.group(2)
                
                # Apply multiplier based on unit
                if unit:
                    unit_lower = unit.lower()
                    if unit_lower in ["billion", "b"]:
                        number *= 1_000_000_000
                    elif unit_lower in ["million", "m"]:
                        number *= 1_000_000
                
                return number
            
            return 0
            
        except Exception as e:
            logger.error(f"Error parsing financial value '{value_text}': {str(e)}")
            return 0
    
    async def analyze_company_sentiment(self, ticker: str, days: int = 30) -> Dict[str, Any]:
        """
        Analyze sentiment for a company
        
        Args:
            ticker: Company ticker symbol
            days: Number of days to analyze
            
        Returns:
            Sentiment analysis
        """
        try:
            # Get news articles for sentiment analysis
            news_articles = await self.market_data_ingestion.get_company_news(ticker, days=days)
            
            # If no news articles, use SEC filings
            if not news_articles:
                sec_filings = await self.sec_edgar_ingestion.get_filings_data(
                    ticker=ticker,
                    filing_types=["8-K"],  # Use 8-K filings for current events
                    limit=5
                )
                
                # Convert to format for sentiment analysis
                for filing in sec_filings:
                    news_articles.append({
                        "ticker": ticker,
                        "headline": f"{filing.get('filing_type')} Filing - {filing.get('filing_date')}",
                        "content": filing.get('content', ''),
                        "datetime": filing.get('filing_date'),
                        "source": "SEC EDGAR",
                        "content_type": "sec_filing"
                    })
            
            # Perform sentiment analysis
            sentiment_results = []
            
            for article in news_articles:
                # Analyze sentiment
                sentiment = await self.sentiment_analyzer.analyze_text(article.get('content', ''))
                
                sentiment_results.append({
                    "headline": article.get('headline', ''),
                    "source": article.get('source', ''),
                    "date": article.get('datetime', ''),
                    "sentiment_score": sentiment,
                    "sentiment_category": "positive" if sentiment > 0.05 else "negative" if sentiment < -0.05 else "neutral"
                })
            
            # Calculate aggregate sentiment
            if sentiment_results:
                sentiment_scores = [result['sentiment_score'] for result in sentiment_results]
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                
                positive_count = sum(1 for s in sentiment_scores if s > 0.05)
                negative_count = sum(1 for s in sentiment_scores if s < -0.05)
                neutral_count = len(sentiment_scores) - positive_count - negative_count
                
                sentiment_distribution = {
                    "positive": positive_count / len(sentiment_scores) * 100,
                    "neutral": neutral_count / len(sentiment_scores) * 100,
                    "negative": negative_count / len(sentiment_scores) * 100
                }
            else:
                avg_sentiment = 0
                sentiment_distribution = {"positive": 0, "neutral": 0, "negative": 0}
            
            # Format response
            result = {
                "ticker": ticker,
                "average_sentiment": avg_sentiment,
                "sentiment_category": "positive" if avg_sentiment > 0.05 else "negative" if avg_sentiment < -0.05 else "neutral",
                "sentiment_distribution": sentiment_distribution,
                "articles_analyzed": len(sentiment_results),
                "period_days": days,
                "detailed_results": sentiment_results[:10],  # Limit to 10 for brevity
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {ticker}: {str(e)}")
            return {
                "ticker": ticker,
                "average_sentiment": 0,
                "sentiment_category": "neutral",
                "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0},
                "articles_analyzed": 0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def start_company_research(self, 
                                  ticker: str,
                                  topics: List[str] = [],
                                  time_period: Optional[str] = None) -> str:
        """
        Start a company research report generation
        
        Args:
            ticker: Company ticker symbol
            topics: List of topics to research
            time_period: Optional time period to focus on
            
        Returns:
            Report ID
        """
        try:
            # Create a new research report
            report_id = str(uuid.uuid4())
            
            # Default topics if none provided
            if not topics:
                topics = ["Financial Performance", "Business Overview", "Risks", "Future Outlook"]
            
            # Create initial report structure
            report = {
                "report_id": report_id,
                "ticker": ticker,
                "topics": topics,
                "time_period": time_period,
                "sections": [],
                "summary": "",
                "sources": [],
                "status": "processing",
                "timestamp": datetime.now().isoformat()
            }
            
            # Store the initial report
            await store_research_report(self.db, report)
            
            return report_id
            
        except Exception as e:
            logger.error(f"Error starting company research for {ticker}: {str(e)}")
            return ""
    
    async def generate_research_report(self, report_id: str) -> bool:
        """
        Generate a company research report
        
        Args:
            report_id: Report ID
            
        Returns:
            Success status
        """
        try:
            # Get the report from the database
            report = await get_research_report(self.db, report_id)
            
            if not report:
                logger.error(f"Research report {report_id} not found")
                return False
            
            # Extract report information
            ticker = report.get("ticker")
            topics = report.get("topics", [])
            time_period = report.get("time_period")
            
            # Generate report sections
            sections = []
            all_sources = []
            
            for topic in topics:
                # Generate query for the topic
                topic_query = f"Provide an analysis of {ticker}'s {topic}"
                if time_period:
                    topic_query += f" for {time_period}"
                
                # Get response
                response = await self.process_query(topic_query, ticker=ticker, expand_query=True)
                
                # Add section
                sections.append({
                    "title": topic,
                    "content": response.get("answer", ""),
                    "sources": response.get("sources", [])
                })
                
                # Collect sources
                all_sources.extend(response.get("sources", []))
            
            # Generate executive summary
            summary_query = f"Provide a concise executive summary of {ticker}"
            if time_period:
                summary_query += f" for {time_period}"
            
            summary_response = await self.process_query(summary_query, ticker=ticker, expand_query=True)
            summary = summary_response.get("answer", "")
            
            # Add summary sources
            all_sources.extend(summary_response.get("sources", []))
            
            # Remove duplicate sources
            unique_sources = []
            source_urls = set()
            
            for source in all_sources:
                source_key = source.get("source", "")
                if source_key not in source_urls:
                    source_urls.add(source_key)
                    unique_sources.append(source)
            
            # Update the report
            update_data = {
                "sections": sections,
                "summary": summary,
                "sources": unique_sources,
                "status": "completed",
                "completed_timestamp": datetime.now().isoformat()
            }
            
            success = await update_research_report(self.db, report_id, update_data)
            
            return success
            
        except Exception as e:
            logger.error(f"Error generating research report {report_id}: {str(e)}")
            
            # Update report with error status
            error_update = {
                "status": "error",
                "error_message": str(e),
                "completed_timestamp": datetime.now().isoformat()
            }
            
            await update_research_report(self.db, report_id, error_update)
            
            return False
    
    async def get_research_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a research report
        
        Args:
            report_id: Report ID
            
        Returns:
            Research report or None if not found
        """
        return await get_research_report(self.db, report_id)
    
    async def get_company_data_summary(self, ticker: str) -> Dict[str, Any]:
        """
        Get a summary of available data for a company
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            Summary of available data
        """
        try:
            # Get company record
            company = await get_company(self.db, ticker)
            
            # Get document statistics
            documents = await list_company_documents(self.db, ticker)
            
            # Count document types
            doc_counts = {
                "sec_filings": 0,
                "news": 0,
                "financial_data": 0,
                "other": 0
            }
            
            for doc in documents:
                content_type = doc.get("content_type", "other")
                if content_type == "sec_filing":
                    doc_counts["sec_filings"] += 1
                elif content_type == "news":
                    doc_counts["news"] += 1
                elif content_type == "financial_data":
                    doc_counts["financial_data"] += 1
                else:
                    doc_counts["other"] += 1
            
            # Get latest stock price data if needed
            latest_price = None
            price_date = None
            
            if not company or "latest_price" not in company:
                try:
                    price_data = await self.market_data_ingestion.get_stock_price_data(ticker, period="1mo")
                    if price_data and len(price_data) > 0:
                        latest_price = price_data[-1].get("close")
                        price_date = price_data[-1].get("date")
                except Exception as e:
                    logger.warning(f"Error getting price data for {ticker}: {str(e)}")
            else:
                latest_price = company.get("latest_price")
                price_date = company.get("latest_price_date")
            
            # Build summary
            summary = {
                "ticker": ticker,
                "document_counts": doc_counts,
                "total_documents": sum(doc_counts.values()),
                "latest_price": latest_price,
                "price_date": price_date,
                "last_updated": company.get("last_updated") if company else None
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting company data summary for {ticker}: {str(e)}")
            return {
                "ticker": ticker,
                "error": str(e),
                "document_counts": {"sec_filings": 0, "news": 0, "financial_data": 0, "other": 0},
                "total_documents": 0
            }