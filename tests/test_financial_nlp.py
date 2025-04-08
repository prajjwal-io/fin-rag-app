

import pytest
import asyncio
from typing import Dict, Any
from app.core.financial_nlp.sentiment_analyzer import SentimentAnalyzer
from app.core.financial_nlp.entity_extractor import EntityExtractor
from app.core.financial_nlp.financial_metrics import FinancialMetricsAnalyzer

pytestmark = pytest.mark.asyncio

class TestSentimentAnalyzer:
    async def test_analyze_text(self, sentiment_analyzer: SentimentAnalyzer):
        """
        Test analyzing sentiment of text.
        """
        # Test positive sentiment
        positive_text = "The company reported strong revenue growth and exceeded analyst expectations."
        positive_sentiment = await sentiment_analyzer.analyze_text(positive_text)
        
        # Test negative sentiment
        negative_text = "The company missed earnings estimates and reported declining sales."
        negative_sentiment = await sentiment_analyzer.analyze_text(negative_text)
        
        # Verify that positive sentiment is higher than negative sentiment
        assert positive_sentiment > negative_sentiment
        
        # Verify that sentiment is within the expected range
        assert -1 <= positive_sentiment <= 1
        assert -1 <= negative_sentiment <= 1

class TestEntityExtractor:
    async def test_extract_entities(self, entity_extractor: EntityExtractor):
        """
        Test extracting entities from text.
        """
        # Test text with financial entities
        text = """
        Apple Inc. (AAPL) reported quarterly revenue of $89.5 billion.
        The company's EPS was $1.46, and gross margin was 45.2%.
        The results were for the quarter ended September 30, 2023.
        """
        
        entities = await entity_extractor.extract_entities(text)
        
        # Verify that we extracted some entities
        assert len(entities["tickers"]) > 0 or len(entities["metrics"]) > 0 or len(entities["amounts"]) > 0 or len(entities["percentages"]) > 0

class TestFinancialMetricsAnalyzer:
    def test_extract_financial_values(self, financial_metrics_analyzer: FinancialMetricsAnalyzer):
        """
        Test extracting financial values from text.
        """
        # Test text with financial values
        text = """
        Apple reported revenue of $89.5 billion, up 8% year over year.
        Net income was $22.96 billion, with a profit margin of 25.7%.
        Earnings per share grew by 13% to $1.46.
        """
        
        values = financial_metrics_analyzer.extract_financial_values(text)
        
        # Verify that we extracted some values
        assert len(values["revenues"]) > 0 or len(values["profits"]) > 0 or len(values["margins"]) > 0 or len(values["growth_rates"]) > 0