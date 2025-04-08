import re
import html
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Clean text from HTML, excess whitespace, and other noise
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    try:
        # Step 1: Decode HTML entities
        text = html.unescape(text)
        
        # Step 2: Remove HTML tags using BeautifulSoup
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text(separator=' ')
        
        # Step 3: Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Step 4: Remove non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        
        # Step 5: Remove special SEC EDGAR formatting
        text = re.sub(r'^\s*<?DOCUMENT>\s*', '', text)
        text = re.sub(r'^\s*<TYPE>\s*', '', text)
        
        # Step 6: Normalize line breaks
        text = re.sub(r'[\r\n]+', '\n', text)
        
        # Step 7: Trim leading/trailing whitespace
        text = text.strip()
        
        return text
        
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        return text  # Return original text if cleaning fails