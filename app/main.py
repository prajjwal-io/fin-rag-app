from fastapi import FastAPI, APIRouter, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from datetime import datetime
import os
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import openai

from .config.settings import settings
from .api.routes import router as api_router
from .db.mongodb import close_db_connection, get_database
from .core.vector_store.pinecone_client import PineconeVectorStore

# Version
__version__ = "1.0.0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Financial Research Copilot with RAG",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {settings.PROJECT_NAME} v{__version__}")
    
    # Test database connection
    try:
        client = AsyncIOMotorClient(settings.MONGODB_URI, serverSelectionTimeoutMS=5000)
        await client.admin.command('ping')
        logger.info("MongoDB connection successful")
    except Exception as e:
        logger.error(f"MongoDB connection failed: {str(e)}")
    
    # Test Pinecone connection if API key provided
    if settings.PINECONE_API_KEY:
        try:
            pinecone_client = PineconeVectorStore()
            if pinecone_client.index:
                logger.info(f"Pinecone index '{settings.PINECONE_INDEX_NAME}' connected")
            else:
                logger.warning(f"Pinecone index '{settings.PINECONE_INDEX_NAME}' not found")
        except Exception as e:
            logger.error(f"Pinecone connection failed: {str(e)}")
    else:
        logger.warning("Pinecone API key not provided")
    
    # Test OpenAI API key if provided
    if settings.OPENAI_API_KEY:
        try:
            openai.api_key = settings.OPENAI_API_KEY
            # Make a minimal API call to verify the key
            models = openai.models.list()
            if models:
                logger.info("OpenAI API key is valid")
        except Exception as e:
            logger.error(f"OpenAI API key validation failed: {str(e)}")
    else:
        logger.warning("OpenAI API key not provided")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"Shutting down {settings.PROJECT_NAME}")
    await close_db_connection()

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "version": __version__,
        "api_docs": "/docs",
        "health_check": "/health"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and load balancers
    """
    try:
        # Check database connection
        client = AsyncIOMotorClient(settings.MONGODB_URI, serverSelectionTimeoutMS=5000)
        await client.admin.command('ping')
        
        # Check Pinecone connection if API key provided
        pinecone_status = "Not Configured"
        if settings.PINECONE_API_KEY:
            try:
                pinecone_client = PineconeVectorStore()
                if pinecone_client.index:
                    pinecone_status = "Connected"
                else:
                    pinecone_status = "Index Not Found"
            except Exception:
                pinecone_status = "Connection Error"
        
        # Check OpenAI connection if API key provided
        openai_status = "Not Configured"
        if settings.OPENAI_API_KEY:
            try:
                # Simple test call
                openai.api_key = settings.OPENAI_API_KEY
                response = openai.embeddings.create(
                    model=settings.EMBEDDING_MODEL,
                    input="test"
                )
                if response and hasattr(response, 'data'):
                    openai_status = "Connected"
                else:
                    openai_status = "Invalid Response"
            except Exception:
                openai_status = "Connection Error"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": __version__,
            "environment": os.environ.get("ENVIRONMENT", "development"),
            "services": {
                "mongodb": "Connected",
                "pinecone": pinecone_status,
                "openai": openai_status
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

# Version endpoint
@app.get("/version")
async def version():
    """
    Get the application version
    """
    return {
        "version": __version__,
        "name": settings.PROJECT_NAME,
        "environment": os.environ.get("ENVIRONMENT", "development")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)