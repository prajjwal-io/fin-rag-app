import re
from datetime import datetime
import logging
from typing import Dict, Any, List, Tuple
import spacy
from textblob import TextBlob

logger = logging.getLogger(__name__)

# Load spaCy model - Need to first download it with: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except:
    logger.warning("spaCy model not found. Please download it using: python -m spacy download en_core_web_sm")
    nlp = None

class MetadataExtractor:
    def extract_financial_periods(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract financial periods (dates, quarters, years) from text
        
        Args:
            text: Input text
            
        Returns:
            List of (period_type, period_value) tuples
        """
        periods = []
        
        # Extract fiscal year references
        fy_pattern = r'(?:fiscal year|FY|F\.Y\.|fiscal|year(?:s)?)[ ]?(?:ended|ending)?[ ]?(?:on)?[ ]?(\d{4})'
        fy_matches = re.finditer(fy_pattern, text, re.IGNORECASE)
        
        for match in fy_matches:
            year = match.group(1)
            periods.append(("fiscal_year", year))
        
        # Extract quarter references
        q_pattern = r'(?:(?:first|second|third|fourth|1st|2nd|3rd|4th|Q1|Q2|Q3|Q4)[ ]?(?:quarter)[ ]?(?:of)?[ ]?(?:fiscal)?[ ]?(?:year)?[ ]?(\d{4}))'
        q_matches = re.finditer(q_pattern, text, re.IGNORECASE)
        
        for match in q_matches:
            quarter_text = match.group(0)
            year = match.group(1)
            
            quarter = None
            if re.search(r'first|1st|Q1', quarter_text, re.IGNORECASE):
                quarter = "Q1"
            elif re.search(r'second|2nd|Q2', quarter_text, re.IGNORECASE):
                quarter = "Q2"
            elif re.search(r'third|3rd|Q3', quarter_text, re.IGNORECASE):
                quarter = "Q3"
            elif re.search(r'fourth|4th|Q4', quarter_text, re.IGNORECASE):
                quarter = "Q4"
            
            if quarter:
                periods.append(("quarter", f"{quarter} {year}"))
        
        # Extract date references
        date_patterns = [
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)[ ]\d{1,2},[ ]\d{4}',
            r'\d{1,2}/\d{1,2}/\d{4}',
            r'\d{4}-\d{2}-\d{2}'
        ]
        
        for pattern in date_patterns:
            date_matches = re.finditer(pattern, text)
            for match in date_matches:
                date_str = match.group(0)
                periods.append(("date", date_str))
        
        return periods
    
    def extract_financial_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract financial entities from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and their values
        """
        if not nlp:
            return {"companies": [], "organizations": [], "financial_terms": []}
        
        entities = {
            "companies": [],
            "organizations": [],
            "financial_terms": []
        }
        
        # Financial terms to look for
        financial_terms = [
            "revenue", "income", "profit", "loss", "earnings", "EBITDA", "EPS", "dividend",
            "assets", "liabilities", "equity", "cash flow", "balance sheet", "income statement",
            "debt", "credit", "investment", "expense", "tax", "depreciation", "amortization",
            "gross margin", "operating margin", "net margin", "ROI", "ROE", "ROA"
        ]
        
        # Process with spaCy
        doc = nlp(text[:10000])  # Limit to first 10,000 chars for performance
        
        # Extract organizations
        for ent in doc.ents:
            if ent.label_ == "ORG":
                if len(ent.text) > 2:  # Filter out short acronyms
                    if any(term in ent.text.lower() for term in ["corp", "inc", "ltd", "company", "corporation"]):
                        entities["companies"].append(ent.text)
                    else:
                        entities["organizations"].append(ent.text)
        
        # Extract financial terms
        for term in financial_terms:
            if term.lower() in text.lower():
                entities["financial_terms"].append(term)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def extract_metadata(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and add metadata to a document
        
        Args:
            document: Input document
            
        Returns:
            Document with extracted metadata
        """
        try:
            content = document.get('content', '')
            if not content:
                return document
            
            # Create metadata object if it doesn't exist
            if 'metadata' not in document:
                document['metadata'] = {}
            
            # Extract financial periods
            periods = self.extract_financial_periods(content)
            document['metadata']['financial_periods'] = periods
            
            # Extract financial entities
            entities = self.extract_financial_entities(content)
            document['metadata']['entities'] = entities
            
            # Extract sentiment (basic)
            if len(content) > 0:
                blob = TextBlob(content[:5000])  # Limit for performance
                document['metadata']['sentiment'] = blob.sentiment.polarity
                document['metadata']['subjectivity'] = blob.sentiment.subjectivity
            
            return document
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            return document