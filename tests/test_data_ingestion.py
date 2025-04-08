import pytest
import asyncio
from typing import Dict, Any
from app.core.data_ingestion.sec_edgar import SECEdgarIngestion
from app.core.data_ingestion.market_data import MarketDataIngestion

pytestmark = pytest.mark.asyncio

class TestSECEdgarIngestion:
    async def test_get_filings_data(self, sec_edgar_ingestion: SECEdgarIngestion):
        """
        Test retrieving filings data from SEC EDGAR.
        """
        # This test is marked as optional since it makes real API calls
        pytest.skip("Optional test that makes real API calls")
        
        # Get filings for a well-known company
        filings = await sec_edgar_ingestion.get_filings_data(
            ticker="AAPL",
            filing_types=["10-K"],
            limit=1
        )
        
        # Verify that we got at least one filing
        assert len(filings) > 0
        
        # Verify the structure of the first filing
        filing = filings[0]
        assert "ticker" in filing
        assert "filing_type" in filing
        assert "filing_date" in filing
        assert "content" in filing
        assert "source" in filing
        
        # Verify filing content
        assert len(filing["content"]) > 0
        assert filing["ticker"] == "AAPL"
        assert filing["filing_type"] == "10-K"

class TestMarketDataIngestion:
    async def test_get_stock_price_data(self, market_data_ingestion: MarketDataIngestion):
        """
        Test retrieving stock price data.
        """
        # This test is marked as optional since it makes real API calls
        pytest.skip("Optional test that makes real API calls")
        
        # Get price data for a well-known company
        price_data = await market_data_ingestion.get_stock_price_data(
            ticker="AAPL",
            period="1mo",
            interval="1d"
        )
        
        # Verify that we got some price data
        assert len(price_data) > 0
        
        # Verify the structure of the price data
        first_day = price_data[0]
        assert "date" in first_day
        assert "open" in first_day
        assert "close" in first_day
        assert "high" in first_day
        assert "low" in first_day
        assert "volume" in first_day
    
    async def test_get_company_financials(self, market_data_ingestion: MarketDataIngestion):
        """
        Test retrieving company financials.
        """
        # This test is marked as optional since it makes real API calls
        pytest.skip("Optional test that makes real API calls")
        
        # Get financials for a well-known company
        financials = await market_data_ingestion.get_company_financials("AAPL")
        
        # Verify that we got some financial data
        assert "income_statement" in financials
        assert "balance_sheet" in financials
        assert "cash_flow" in financials


