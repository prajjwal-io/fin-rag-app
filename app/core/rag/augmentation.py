from typing import List, Dict, Any, Optional
import logging
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from ...config.settings import settings

logger = logging.getLogger(__name__)

class QueryAugmentation:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            temperature=0.2
        )
    
    async def expand_financial_query(self, query: str) -> str:
        """
        Expand a financial query to improve retrieval
        
        Args:
            query: Original user query
            
        Returns:
            Expanded query
        """
        try:
            # Create prompt template for query expansion
            prompt_template = PromptTemplate.from_template(
                """You are an expert in financial analysis and investment research.
                
                The following is a user query related to financial analysis or investment research:
                
                USER QUERY: {query}
                
                Please expand this query to include relevant financial terms, metrics, and concepts that would help in retrieving better search results. Your expansion should maintain the original intent of the query while making it more comprehensive for a vector search system.
                
                EXPANDED QUERY:"""
            )
            
            # Create and run the LLM chain
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            expanded_query = chain.run(query=query)
            
            logger.info(f"Expanded query: {expanded_query}")
            return expanded_query.strip()
            
        except Exception as e:
            logger.error(f"Error expanding query: {str(e)}")
            return query
    
    async def generate_search_filters(self, query: str) -> Dict[str, Any]:
        """
        Generate search filters based on query content
        
        Args:
            query: User query
            
        Returns:
            Dictionary of search filters
        """
        try:
            # Create prompt template for filter generation
            prompt_template = PromptTemplate.from_template(
                """You are an expert in financial analysis and investment research.
                
                The following is a user query related to financial analysis or investment research:
                
                USER QUERY: {query}
                
                Based on this query, identify the following elements (if present):
                1. Company ticker symbols or names
                2. Time periods or dates
                3. Financial document types (10-K, 10-Q, 8-K, earnings call, etc.)
                4. Financial metrics or KPIs
                
                Return your analysis as a JSON object with these keys: "tickers", "time_periods", "document_types", "metrics".
                If any element is not present, use an empty list for that key. Format as valid JSON only.
                
                JSON RESPONSE:"""
            )
            
            # Create and run the LLM chain
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            filter_json = chain.run(query=query)
            
            # Clean and parse the JSON
            filter_json = filter_json.strip()
            
            # Handle potential markdown code block formatting
            if filter_json.startswith("```json"):
                filter_json = filter_json.replace("```json", "").replace("```", "").strip()
            elif filter_json.startswith("```"):
                filter_json = filter_json.replace("```", "").strip()
            
            filters = json.loads(filter_json)
            
            # Convert to Pinecone filter format
            pinecone_filters = {}
            
            if filters.get("tickers"):
                if len(filters["tickers"]) == 1:
                    pinecone_filters["ticker"] = filters["tickers"][0]
                elif len(filters["tickers"]) > 1:
                    pinecone_filters["ticker"] = {"$in": filters["tickers"]}
            
            if filters.get("document_types"):
                if len(filters["document_types"]) == 1:
                    pinecone_filters["content_type"] = filters["document_types"][0]
                elif len(filters["document_types"]) > 1:
                    pinecone_filters["content_type"] = {"$in": filters["document_types"]}
            
            logger.info(f"Generated search filters: {pinecone_filters}")
            return pinecone_filters
            
        except Exception as e:
            logger.error(f"Error generating search filters: {str(e)}")
            return {}
    
    async def extract_financial_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract financial entities from a query
        
        Args:
            query: User query
            
        Returns:
            Dictionary of extracted entities
        """
        try:
            # Create prompt template for entity extraction
            prompt_template = PromptTemplate.from_template(
                """You are an expert in financial analysis and investment research.
                
                The following is a user query related to financial analysis or investment research:
                
                USER QUERY: {query}
                
                Extract all financial entities from this query and categorize them. Return your extraction as a JSON object with these keys:
                - "companies": List of company names/tickers
                - "metrics": List of financial metrics mentioned
                - "time_periods": List of time periods/dates mentioned
                - "financial_terms": List of financial terms/concepts
                
                Format as valid JSON only.
                
                JSON RESPONSE:"""
            )
            
            # Create and run the LLM chain
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            entities_json = chain.run(query=query)
            
            # Clean and parse the JSON
            entities_json = entities_json.strip()
            
            # Handle potential markdown code block formatting
            if entities_json.startswith("```json"):
                entities_json = entities_json.replace("```json", "").replace("```", "").strip()
            elif entities_json.startswith("```"):
                entities_json = entities_json.replace("```", "").strip()
            
            entities = json.loads(entities_json)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting financial entities: {str(e)}")
            return {
                "companies": [],
                "metrics": [],
                "time_periods": [],
                "financial_terms": []
            }