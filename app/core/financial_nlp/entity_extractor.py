import logging
from typing import Dict, Any, List, Union
import re
import openai
import spacy
import json
from ...config.settings import settings

logger = logging.getLogger(__name__)

# Load spaCy model - Need to first download it with: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except:
    logger.warning("spaCy model not found. Please download it using: python -m spacy download en_core_web_sm")
    nlp = None

class EntityExtractor:
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.confidence = settings.ENTITY_CONFIDENCE
        openai.api_key = self.api_key
        
        # Common financial metric patterns
        self.metric_patterns = {
            "revenue": r"(?:total |annual |quarterly )?(?:revenue|sales)(?:\s+of\s+[\$]?[\d\.]+\s+(?:million|billion|trillion|M|B|T))?",
            "profit": r"(?:net |gross |operating )?(?:profit|income|earnings)(?:\s+of\s+[\$]?[\d\.]+\s+(?:million|billion|trillion|M|B|T))?",
            "eps": r"(?:EPS|earnings per share)(?:\s+of\s+[\$]?[\d\.]+)?",
            "growth": r"(?:revenue |sales |profit |income )?growth(?:\s+of\s+[\d\.]+\%)?",
            "margin": r"(?:gross |operating |net |profit )?margin(?:\s+of\s+[\d\.]+\%)?",
        }
    
    async def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract financial entities from text
        
        Args:
            text: Financial text
            
        Returns:
            Dictionary of entity types and their values
        """
        if not nlp:
            # Fallback to basic regex if spaCy not available
            return self._extract_entities_regex(text)
        
        try:
            entities = {
                "companies": [],
                "tickers": [],
                "metrics": [],
                "dates": [],
                "amounts": [],
                "percentages": []
            }
            
            # Process with spaCy
            doc = nlp(text[:10000])  # Limit to first 10,000 chars for performance
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    entities["companies"].append(ent.text)
                elif ent.label_ == "DATE":
                    entities["dates"].append(ent.text)
                elif ent.label_ == "MONEY" or ent.label_ == "CARDINAL":
                    if "$" in ent.text or "dollar" in ent.text.lower():
                        entities["amounts"].append(ent.text)
                    elif "%" in ent.text or "percent" in ent.text.lower():
                        entities["percentages"].append(ent.text)
            
            # Extract stock tickers (uppercase 1-5 letter words)
            ticker_pattern = r'\b[A-Z]{1,5}\b'
            tickers = re.findall(ticker_pattern, text)
            # Filter out common words in all caps
            common_words = ["A", "I", "CEO", "CFO", "COO", "CTO", "Q", "K"]
            tickers = [t for t in tickers if t not in common_words]
            entities["tickers"].extend(tickers)
            
            # Extract financial metrics
            for metric_name, pattern in self.metric_patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    entities["metrics"].extend(matches)
            
            # Remove duplicates
            for key in entities:
                entities[key] = list(set(entities[key]))
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return self._extract_entities_regex(text)
    
    def _extract_entities_regex(self, text: str) -> Dict[str, List[str]]:
        """
        Fallback entity extraction using regex patterns
        
        Args:
            text: Financial text
            
        Returns:
            Dictionary of entity types and their values
        """
        entities = {
            "companies": [],
            "tickers": [],
            "metrics": [],
            "dates": [],
            "amounts": [],
            "percentages": []
        }
        
        # Extract tickers (uppercase 1-5 letter words)
        ticker_pattern = r'\b[A-Z]{1,5}\b'
        tickers = re.findall(ticker_pattern, text)
        # Filter out common words in all caps
        common_words = ["A", "I", "CEO", "CFO", "COO", "CTO", "Q", "K"]
        tickers = [t for t in tickers if t not in common_words]
        entities["tickers"].extend(tickers)
        
        # Extract financial metrics
        for metric_name, pattern in self.metric_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities["metrics"].extend(matches)
        
        # Extract dollar amounts
        amount_pattern = r'[\$]?[\d,]+\.?\d*\s+(?:million|billion|trillion|M|B|T)'
        amounts = re.findall(amount_pattern, text)
        entities["amounts"].extend(amounts)
        
        # Extract percentages
        percent_pattern = r'[\d\.]+\%'
        percentages = re.findall(percent_pattern, text)
        entities["percentages"].extend(percentages)
        
        # Extract dates
        date_pattern = r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,?\s+\d{4})?\b'
        dates = re.findall(date_pattern, text)
        entities["dates"].extend(dates)
        
        date_pattern2 = r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b'
        dates2 = re.findall(date_pattern2, text)
        entities["dates"].extend(dates2)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    async def extract_with_llm(self, text: str) -> Dict[str, List[str]]:
        """
        Extract financial entities using OpenAI
        
        Args:
            text: Financial text
            
        Returns:
            Dictionary of entity types and their values
        """
        try:
            # Create prompt for entity extraction
            prompt = f"""Extract all financial entities from the following text. 
            
            Text: {text[:4000]}
            
            Extract these entity types:
            1. Companies (company names)
            2. Tickers (stock symbols)
            3. Financial metrics (revenue, profit, EPS, margins, etc.)
            4. Dates and time periods
            5. Financial amounts (dollar values)
            6. Percentages
            
            Format the response as JSON with these keys: "companies", "tickers", "metrics", "dates", "amounts", "percentages".
            Each key should map to a list of extracted entities.
            """
            
            # Call OpenAI API
            response = openai.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a financial entity extraction expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting entities with LLM: {str(e)}")
            # Fallback to spaCy/regex extraction
            return await self.extract_entities(text)