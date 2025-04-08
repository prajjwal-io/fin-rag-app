import logging
from typing import Dict, Any, List, Union
from textblob import TextBlob
import re
import numpy as np
import openai
from ...config.settings import settings

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.threshold = settings.SENTIMENT_THRESHOLD
        openai.api_key = self.api_key
        
        # Financial sentiment words
        self.positive_words = [
            "growth", "profit", "increase", "exceed", "outperform", "beat", "strong", "success", 
            "positive", "gain", "improve", "opportunity", "upside", "optimistic", "advantage",
            "favorable", "robust", "momentum", "efficiently", "confidence", "progress"
        ]
        
        self.negative_words = [
            "decline", "loss", "decrease", "miss", "underperform", "weak", "fail", "negative",
            "risk", "concern", "challenge", "downside", "pessimistic", "disadvantage", "unfavorable",
            "volatile", "uncertainty", "inefficiently", "doubt", "delay", "struggle", "liability"
        ]
    
    async def analyze_text(self, text: str) -> float:
        """
        Analyze sentiment of financial text
        
        Args:
            text: Financial text
            
        Returns:
            Sentiment score (-1 to 1)
        """
        try:
            # Basic sentiment analysis with TextBlob
            blob = TextBlob(text)
            basic_sentiment = blob.sentiment.polarity
            
            # Count financial sentiment words
            text_lower = text.lower()
            positive_count = sum(text_lower.count(word) for word in self.positive_words)
            negative_count = sum(text_lower.count(word) for word in self.negative_words)
            
            # Calculate financial sentiment adjustment
            if positive_count + negative_count > 0:
                financial_sentiment = (positive_count - negative_count) / (positive_count + negative_count)
            else:
                financial_sentiment = 0
            
            # Combine sentiments (weighted average)
            combined_sentiment = (basic_sentiment * 0.4) + (financial_sentiment * 0.6)
            
            return combined_sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return 0
    
    async def analyze_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment of a document
        
        Args:
            document: Document with content
            
        Returns:
            Document with sentiment analysis
        """
        try:
            content = document.get('content', '')
            if not content:
                return document
            
            # Analyze sentiment
            sentiment = await self.analyze_text(content)
            
            # Add sentiment to document
            if 'metadata' not in document:
                document['metadata'] = {}
            
            document['metadata']['sentiment'] = sentiment
            
            # Add sentiment classification
            if sentiment >= self.threshold:
                document['metadata']['sentiment_classification'] = 'positive'
            elif sentiment <= -self.threshold:
                document['metadata']['sentiment_classification'] = 'negative'
            else:
                document['metadata']['sentiment_classification'] = 'neutral'
            
            return document
            
        except Exception as e:
            logger.error(f"Error analyzing document sentiment: {str(e)}")
            return document
    
    async def analyze_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch analyze sentiment of multiple documents
        
        Args:
            documents: List of documents
            
        Returns:
            Documents with sentiment analysis
        """
        analyzed_documents = []
        
        for doc in documents:
            analyzed_doc = await self.analyze_document(doc)
            analyzed_documents.append(analyzed_doc)
        
        return analyzed_documents
    
    async def analyze_sentiment_with_llm(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using OpenAI for more nuanced analysis
        
        Args:
            text: Financial text
            
        Returns:
            Dictionary with sentiment analysis
        """
        try:
            # Create prompt for sentiment analysis
            prompt = f"""Analyze the sentiment of the following financial text. 
            Consider financial terms and context specifically.
            
            Text: {text[:4000]}
            
            Rate the sentiment on a scale from -1.0 (very negative) to 1.0 (very positive).
            Identify key positive and negative factors.
            Format the response as JSON with these keys: "sentiment_score", "positive_factors", "negative_factors", "confidence".
            """
            
            # Call OpenAI API
            response = openai.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a financial sentiment analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment with LLM: {str(e)}")
            return {
                "sentiment_score": 0,
                "positive_factors": [],
                "negative_factors": [],
                "confidence": 0
            }