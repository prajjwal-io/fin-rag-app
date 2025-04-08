import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from ...config.settings import settings

logger = logging.getLogger(__name__)

class MarketDataIngestion:
    def __init__(self):
        self.alpha_vantage_api_key = settings.ALPHA_VANTAGE_API_KEY
        self.finnhub_api_key = settings.FINNHUB_API_KEY
    
    async def get_stock_price_data(self, ticker: str, period: str = "1y", interval: str = "1d"):
        """
        Get historical stock price data using yfinance
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with stock price data
        """
        try:
            logger.info(f"Getting stock price data for {ticker} over {period} with {interval} interval")
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            
            # Process DataFrame
            df.reset_index(inplace=True)
            df.rename(columns={"Date": "date", "Open": "open", "High": "high", 
                              "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
            
            # Convert date to string format
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")
            
            return df.to_dict(orient="records")
            
        except Exception as e:
            logger.error(f"Error getting stock price data for {ticker}: {str(e)}")
            return []
    
    async def get_company_financials(self, ticker: str):
        """
        Get company financial statements using yfinance
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with income statement, balance sheet, and cash flow data
        """
        try:
            logger.info(f"Getting financial statements for {ticker}")
            stock = yf.Ticker(ticker)
            
            # Get financial statements
            income_stmt = stock.income_stmt
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            
            # Process DataFrames
            financials = {
                "income_statement": income_stmt.reset_index().to_dict(orient="records") if not income_stmt.empty else [],
                "balance_sheet": balance_sheet.reset_index().to_dict(orient="records") if not balance_sheet.empty else [],
                "cash_flow": cash_flow.reset_index().to_dict(orient="records") if not cash_flow.empty else []
            }
            
            return financials
            
        except Exception as e:
            logger.error(f"Error getting financial statements for {ticker}: {str(e)}")
            return {"income_statement": [], "balance_sheet": [], "cash_flow": []}
    
    async def get_company_news(self, ticker: str, days: int = 7):
        """
        Get recent company news using Finnhub API
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back
            
        Returns:
            List of news articles
        """
        if not self.finnhub_api_key:
            logger.warning("Finnhub API key not provided, skipping news retrieval")
            return []
            
        try:
            logger.info(f"Getting news for {ticker} over the past {days} days")
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Format dates for API
            from_date = start_date.strftime("%Y-%m-%d")
            to_date = end_date.strftime("%Y-%m-%d")
            
            # Make API request
            url = f"https://finnhub.io/api/v1/company-news"
            params = {
                "symbol": ticker,
                "from": from_date,
                "to": to_date,
                "token": self.finnhub_api_key
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                news = response.json()
                
                # Process news articles
                processed_news = []
                for article in news:
                    processed_news.append({
                        "ticker": ticker,
                        "headline": article.get("headline", ""),
                        "summary": article.get("summary", ""),
                        "url": article.get("url", ""),
                        "datetime": datetime.fromtimestamp(article.get("datetime", 0)).strftime("%Y-%m-%d %H:%M:%S"),
                        "source": article.get("source", ""),
                        "content": article.get("summary", ""),  # Use summary as content
                        "content_type": "news"
                    })
                
                return processed_news
            else:
                logger.error(f"Error getting news for {ticker}: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting news for {ticker}: {str(e)}")
            return []
    
    async def format_financial_data_for_embedding(self, ticker: str):
        """
        Format financial data into documents for embedding
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of document dictionaries
        """
        documents = []
        
        # Get financial data
        financials = await self.get_company_financials(ticker)
        
        # Format income statement
        if financials["income_statement"]:
            income_text = f"Income Statement for {ticker}:\n\n"
            for item in financials["income_statement"]:
                for key, value in item.items():
                    if key != "index":
                        income_text += f"{key}: {value}\n"
            
            documents.append({
                "ticker": ticker,
                "content": income_text,
                "content_type": "financial_data",
                "source": "yfinance_income_statement",
                "filing_date": datetime.now().strftime("%Y-%m-%d")
            })
        
        # Format balance sheet
        if financials["balance_sheet"]:
            balance_text = f"Balance Sheet for {ticker}:\n\n"
            for item in financials["balance_sheet"]:
                for key, value in item.items():
                    if key != "index":
                        balance_text += f"{key}: {value}\n"
            
            documents.append({
                "ticker": ticker,
                "content": balance_text,
                "content_type": "financial_data",
                "source": "yfinance_balance_sheet",
                "filing_date": datetime.now().strftime("%Y-%m-%d")
            })
        
        # Format cash flow
        if financials["cash_flow"]:
            cash_flow_text = f"Cash Flow Statement for {ticker}:\n\n"
            for item in financials["cash_flow"]:
                for key, value in item.items():
                    if key != "index":
                        cash_flow_text += f"{key}: {value}\n"
            
            documents.append({
                "ticker": ticker,
                "content": cash_flow_text,
                "content_type": "financial_data",
                "source": "yfinance_cash_flow",
                "filing_date": datetime.now().strftime("%Y-%m-%d")
            })
        
        # Get and format stock price data
        price_data = await self.get_stock_price_data(ticker, period="1y")
        if price_data:
            price_text = f"Stock Price Data for {ticker} (Past Year):\n\n"
            for day in price_data:
                price_text += f"Date: {day.get('date')}, Open: {day.get('open')}, Close: {day.get('close')}, High: {day.get('high')}, Low: {day.get('low')}, Volume: {day.get('volume')}\n"
            
            documents.append({
                "ticker": ticker,
                "content": price_text,
                "content_type": "financial_data",
                "source": "yfinance_price_data",
                "filing_date": datetime.now().strftime("%Y-%m-%d")
            })
        
        return documents