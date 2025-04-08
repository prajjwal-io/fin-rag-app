import os
import sys
import json
from dotenv import load_dotenv, set_key, find_dotenv
import argparse
import logging
import requests
import openai
from pinecone import Pinecone
# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

def test_openai_key(api_key: str) -> bool:
    """
    Test if an OpenAI API key is valid
    
    Args:
        api_key: OpenAI API key
        
    Returns:
        Whether the key is valid
    """
    try:
        
        openai.api_key = api_key
        response = openai.embeddings.create(
            model="text-embedding-3-large",
            input="test"
        )
        return True
    except Exception as e:
        logger.error(f"Error testing OpenAI API key: {str(e)}")
        return False

def test_pinecone_key(api_key: str, environment: str) -> bool:
    """
    Test if a Pinecone API key is valid
    
    Args:
        api_key: Pinecone API key
        environment: Pinecone environment
        
    Returns:
        Whether the key is valid
    """
    try:
        
        pc = Pinecone(api_key=api_key)
        indexes = pc.list_indexes()
        return True
    except Exception as e:
        logger.error(f"Error testing Pinecone API key: {str(e)}")
        return False

def test_finnhub_key(api_key: str) -> bool:
    """
    Test if a Finnhub API key is valid
    
    Args:
        api_key: Finnhub API key
        
    Returns:
        Whether the key is valid
    """
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={api_key}"
        response = requests.get(url)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error testing Finnhub API key: {str(e)}")
        return False

def test_alpha_vantage_key(api_key: str) -> bool:
    """
    Test if an Alpha Vantage API key is valid
    
    Args:
        api_key: Alpha Vantage API key
        
    Returns:
        Whether the key is valid
    """
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        return "Global Quote" in data
    except Exception as e:
        logger.error(f"Error testing Alpha Vantage API key: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Manage API keys for Financial Research Copilot")
    
    # Commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Set command
    set_parser = subparsers.add_parser("set", help="Set an API key")
    set_parser.add_argument("service", choices=["openai", "pinecone", "finnhub", "alpha_vantage"], help="Service name")
    set_parser.add_argument("api_key", help="API key")
    set_parser.add_argument("--environment", help="Environment (for Pinecone)")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test API keys")
    test_parser.add_argument("service", choices=["openai", "pinecone", "finnhub", "alpha_vantage", "all"], help="Service name")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List configured API keys")
    
    args = parser.parse_args()
    
    # Load environment variables
    dotenv_path = find_dotenv()
    if not dotenv_path:
        dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        logger.info(f"Creating new .env file at: {dotenv_path}")
        with open(dotenv_path, "a"):
            pass
    
    load_dotenv(dotenv_path)
    
    if args.command == "set":
        # Set API key
        if args.service == "openai":
            set_key(dotenv_path, "OPENAI_API_KEY", args.api_key)
            logger.info("OpenAI API key set")
            
            # Test key
            if test_openai_key(args.api_key):
                logger.info("OpenAI API key is valid")
            else:
                logger.warning("OpenAI API key is invalid")
        
        elif args.service == "pinecone":
            set_key(dotenv_path, "PINECONE_API_KEY", args.api_key)
            logger.info("Pinecone API key set")
            
            if args.environment:
                set_key(dotenv_path, "PINECONE_ENVIRONMENT", args.environment)
                logger.info(f"Pinecone environment set to: {args.environment}")
            
            # Test key
            if args.environment and test_pinecone_key(args.api_key, args.environment):
                logger.info("Pinecone API key is valid")
            elif not args.environment:
                logger.warning("Pinecone environment not specified, skipping validation")
            else:
                logger.warning("Pinecone API key is invalid")
        
        elif args.service == "finnhub":
            set_key(dotenv_path, "FINNHUB_API_KEY", args.api_key)
            logger.info("Finnhub API key set")
            
            # Test key
            if test_finnhub_key(args.api_key):
                logger.info("Finnhub API key is valid")
            else:
                logger.warning("Finnhub API key is invalid")
        
        elif args.service == "alpha_vantage":
            set_key(dotenv_path, "ALPHA_VANTAGE_API_KEY", args.api_key)
            logger.info("Alpha Vantage API key set")
            
            # Test key
            if test_alpha_vantage_key(args.api_key):
                logger.info("Alpha Vantage API key is valid")
            else:
                logger.warning("Alpha Vantage API key is invalid")
    
    elif args.command == "test":
        # Test API keys
        if args.service == "openai" or args.service == "all":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                if test_openai_key(api_key):
                    logger.info("OpenAI API key is valid")
                else:
                    logger.warning("OpenAI API key is invalid")
            else:
                logger.warning("OpenAI API key not set")
        
        if args.service == "pinecone" or args.service == "all":
            api_key = os.getenv("PINECONE_API_KEY")
            environment = os.getenv("PINECONE_ENVIRONMENT")
            if api_key and environment:
                if test_pinecone_key(api_key, environment):
                    logger.info("Pinecone API key is valid")
                else:
                    logger.warning("Pinecone API key is invalid")
            else:
                logger.warning("Pinecone API key or environment not set")
        
        if args.service == "finnhub" or args.service == "all":
            api_key = os.getenv("FINNHUB_API_KEY")
            if api_key:
                if test_finnhub_key(api_key):
                    logger.info("Finnhub API key is valid")
                else:
                    logger.warning("Finnhub API key is invalid")
            else:
                logger.warning("Finnhub API key not set")
        
        if args.service == "alpha_vantage" or args.service == "all":
            api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
            if api_key:
                if test_alpha_vantage_key(api_key):
                    logger.info("Alpha Vantage API key is valid")
                else:
                    logger.warning("Alpha Vantage API key is invalid")
            else:
                logger.warning("Alpha Vantage API key not set")
    
    elif args.command == "list":
        # List API keys (masked for security)
        openai_key = os.getenv("OPENAI_API_KEY", "")
        pinecone_key = os.getenv("PINECONE_API_KEY", "")
        pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "")
        finnhub_key = os.getenv("FINNHUB_API_KEY", "")
        alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        
        def mask_key(key):
            if not key:
                return "Not set"
            if len(key) <= 8:
                return "*" * len(key)
            return key[:4] + "*" * (len(key) - 8) + key[-4:]
        
        print("\nConfigured API Keys:")
        print(f"OpenAI API Key: {mask_key(openai_key)}")
        print(f"Pinecone API Key: {mask_key(pinecone_key)}")
        print(f"Pinecone Environment: {pinecone_env if pinecone_env else 'Not set'}")
        print(f"Finnhub API Key: {mask_key(finnhub_key)}")
        print(f"Alpha Vantage API Key: {mask_key(alpha_vantage_key)}")
        print("")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()