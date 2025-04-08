from typing import List, Dict, Any, Optional
import logging
import json
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from ...config.settings import settings
from .retriever import DocumentRetriever

logger = logging.getLogger(__name__)

class RAGQueryEngine:
    def __init__(self):
        self.retriever = DocumentRetriever()
        self.llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            temperature=0.1
        )
    
    def format_retrieved_documents(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents for the prompt
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant information found."
        
        context_parts = []
        
        for i, doc in enumerate(documents):
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            
            # Extract metadata fields
            source = metadata.get('source', 'Unknown source')
            doc_type = metadata.get('content_type', 'document')
            filing_type = metadata.get('filing_type', '')
            filing_date = metadata.get('filing_date', '')
            
            # Format document header based on type
            header = f"Document {i+1}"
            if doc_type == "sec_filing" and filing_type:
                header = f"{filing_type} Filing"
                if filing_date:
                    header += f" ({filing_date})"
            elif doc_type == "news":
                header = f"News Article"
                if filing_date:
                    header += f" ({filing_date})"
            elif doc_type == "financial_data":
                header = f"Financial Data"
                if filing_date:
                    header += f" ({filing_date})"
            
            # Format context part
            context_part = f"[{header}]\n{content}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    async def answer_question(self, 
                             query: str, 
                             ticker: Optional[str] = None,
                             content_types: Optional[List[str]] = None,
                             max_tokens: int = 2000) -> Dict[str, Any]:
        """
        Answer a financial question using RAG
        
        Args:
            query: User's question
            ticker: Optional ticker symbol to focus on
            content_types: Optional list of content types to consider
            max_tokens: Maximum tokens for the response
            
        Returns:
            Response dictionary with answer and sources
        """
        try:
            # Retrieve relevant documents
            if ticker:
                documents = await self.retriever.retrieve_by_ticker(query, ticker, content_types)
            else:
                documents = await self.retriever.retrieve_documents(query)
            
            if not documents:
                return {
                    "answer": "I couldn't find relevant information to answer your question. Please try a different question or provide more specific details.",
                    "sources": []
                }
            
            # Format context
            context = self.format_retrieved_documents(documents)
            
            # Create prompt template
            prompt_template = PromptTemplate.from_template(
                """You are a financial research assistant with expertise in analyzing financial documents, SEC filings, and market data.
                
                Answer the following query based ONLY on the provided context information. If the context doesn't contain the information needed to answer the query, say "I don't have enough information to answer this question" and suggest what else might be needed.
                
                CONTEXT:
                {context}
                
                QUERY: {query}
                
                ANSWER:"""
            )
            
            # Create and run the LLM chain
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            answer = chain.run(context=context, query=query)
            
            # Extract sources for citation
            sources = []
            for doc in documents:
                if 'metadata' in doc:
                    metadata = doc['metadata']
                    source = {
                        'type': metadata.get('content_type', 'document'),
                        'source': metadata.get('source', 'Unknown'),
                    }
                    
                    if 'filing_type' in metadata:
                        source['filing_type'] = metadata['filing_type']
                    if 'filing_date' in metadata:
                        source['filing_date'] = metadata['filing_date']
                    
                    sources.append(source)
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error in RAG query engine: {str(e)}")
            return {
                "answer": "I encountered an error while processing your question. Please try again.",
                "sources": []
            }
    
    async def analyze_financial_metrics(self,
                                      ticker: str,
                                      metric_type: str,
                                      time_period: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze specific financial metrics for a company
        
        Args:
            ticker: Company ticker symbol
            metric_type: Type of metric (revenue, profit, growth, etc.)
            time_period: Optional time period to analyze
            
        Returns:
            Analysis result
        """
        try:
            # Build query based on metric type
            if metric_type.lower() == "revenue":
                query = f"What was {ticker}'s revenue"
                if time_period:
                    query += f" in {time_period}"
                query += "? Include growth rates and trends."
            elif metric_type.lower() == "profit" or metric_type.lower() == "earnings":
                query = f"What was {ticker}'s profit or earnings"
                if time_period:
                    query += f" in {time_period}"
                query += "? Include net income, EPS, and profit margins."
            elif metric_type.lower() == "growth":
                query = f"What is {ticker}'s growth rate"
                if time_period:
                    query += f" in {time_period}"
                query += "? Include revenue growth, profit growth, and market expansion."
            else:
                query = f"Analyze {ticker}'s {metric_type}"
                if time_period:
                    query += f" in {time_period}"
            
            # Use existing answer_question method with focused query
            content_types = ["sec_filing", "financial_data", "news"]
            return await self.answer_question(query, ticker, content_types)
            
        except Exception as e:
            logger.error(f"Error analyzing financial metrics: {str(e)}")
            return {
                "answer": f"I encountered an error while analyzing {metric_type} for {ticker}.",
                "sources": []
            }