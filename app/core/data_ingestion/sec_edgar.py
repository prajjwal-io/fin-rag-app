import os
import tempfile
from sec_edgar_downloader import Downloader
from datetime import datetime, timedelta
import logging
from ...config.settings import settings
from ..document_processing.text_cleaner import clean_text

logger = logging.getLogger(__name__)

class SECEdgarIngestion:
    def __init__(self):
        self.downloader = Downloader(user_agent=settings.SEC_USER_AGENT)
        self.temp_dir = tempfile.TemporaryDirectory()
        
    def __del__(self):
        self.temp_dir.cleanup()
    
    async def download_filings(self, ticker: str, filing_type: str, limit: int = 5):
        """
        Download SEC filings for a specific company
        
        Args:
            ticker: Company ticker symbol
            filing_type: Type of filing (10-K, 10-Q, 8-K, etc.)
            limit: Maximum number of filings to download
            
        Returns:
            List of file paths to downloaded filings
        """
        try:
            logger.info(f"Downloading {filing_type} filings for {ticker}")
            
            # Change directory to temp directory for downloads
            original_dir = os.getcwd()
            os.chdir(self.temp_dir.name)
            
            # Download filings
            self.downloader.get(filing_type, ticker, limit=limit, download_details=True)
            
            # Get paths to downloaded filings
            filing_dir = os.path.join(self.temp_dir.name, f"sec-edgar-filings/{ticker}/{filing_type}")
            filings = []
            
            if os.path.exists(filing_dir):
                for folder in sorted(os.listdir(filing_dir), reverse=True):
                    if os.path.isdir(os.path.join(filing_dir, folder)):
                        for file in os.listdir(os.path.join(filing_dir, folder)):
                            if file.endswith(".txt") or file.endswith(".html"):
                                filings.append(os.path.join(filing_dir, folder, file))
            
            # Return to original directory
            os.chdir(original_dir)
            
            logger.info(f"Downloaded {len(filings)} {filing_type} filings for {ticker}")
            return filings
            
        except Exception as e:
            logger.error(f"Error downloading SEC filings for {ticker}: {str(e)}")
            os.chdir(original_dir)
            return []
    
    async def extract_text_from_filing(self, filing_path: str):
        """
        Extract text content from a filing
        
        Args:
            filing_path: Path to the filing file
            
        Returns:
            Extracted text content
        """
        try:
            with open(filing_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # Extract relevant parts (remove headers, etc.)
            if filing_path.endswith(".txt"):
                # Find the start of the actual filing (after the SEC header)
                start_idx = content.find("<DOCUMENT>")
                if start_idx != -1:
                    content = content[start_idx:]
            
            # Clean the text
            cleaned_text = clean_text(content)
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error extracting text from filing {filing_path}: {str(e)}")
            return ""
    
    async def get_filings_data(self, ticker: str, filing_types: list = ["10-K", "10-Q"], limit: int = 3):
        """
        Get text data from SEC filings for a company
        
        Args:
            ticker: Company ticker symbol
            filing_types: Types of filings to download
            limit: Maximum number of filings per type
            
        Returns:
            List of dictionaries with filing data
        """
        filings_data = []
        
        for filing_type in filing_types:
            filing_paths = await self.download_filings(ticker, filing_type, limit)
            
            for path in filing_paths:
                text = await self.extract_text_from_filing(path)
                
                if text:
                    # Extract filing date from path
                    date_parts = path.split('/')[-2].split('-')
                    if len(date_parts) >= 3:
                        filing_date = f"{date_parts[0]}-{date_parts[1]}-{date_parts[2]}"
                    else:
                        filing_date = datetime.now().strftime("%Y-%m-%d")
                    
                    filings_data.append({
                        "ticker": ticker,
                        "filing_type": filing_type,
                        "filing_date": filing_date,
                        "content": text,
                        "source": path,
                        "content_type": "sec_filing"
                    })
        
        return filings_data