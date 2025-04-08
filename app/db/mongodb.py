import logging
from typing import Dict, List, Any, Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from ..config.settings import settings

logger = logging.getLogger(__name__)

# Global database connection instance
_db_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None

async def get_database() -> AsyncIOMotorDatabase:
    """
    Get MongoDB database connection
    
    Returns:
        MongoDB database connection
    """
    global _db_client, _db
    
    if _db is None:
        try:
            # Create MongoDB client
            logger.info(f"Connecting to MongoDB at {settings.MONGODB_URI}")
            _db_client = AsyncIOMotorClient(settings.MONGODB_URI)
            _db = _db_client[settings.MONGODB_DB_NAME]
            
            # Check connection
            await _db_client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
            
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {str(e)}")
            raise
    
    return _db

# MongoDB Collections
COMPANIES_COLLECTION = "companies"
DOCUMENTS_COLLECTION = "documents"
RESEARCH_REPORTS_COLLECTION = "research_reports"
QUERY_HISTORY_COLLECTION = "query_history"
INGEST_HISTORY_COLLECTION = "ingest_history"

async def close_db_connection():
    """
    Close MongoDB connection
    """
    global _db_client
    if _db_client:
        _db_client.close()
        logger.info("MongoDB connection closed")

# Company-related database operations
async def get_company(db: AsyncIOMotorDatabase, ticker: str) -> Optional[Dict[str, Any]]:
    """
    Get company data from database
    
    Args:
        db: MongoDB database connection
        ticker: Company ticker symbol
    
    Returns:
        Company data or None if not found
    """
    return await db[COMPANIES_COLLECTION].find_one({"ticker": ticker})

async def upsert_company(db: AsyncIOMotorDatabase, company_data: Dict[str, Any]) -> str:
    """
    Insert or update company data
    
    Args:
        db: MongoDB database connection
        company_data: Company data to insert or update
    
    Returns:
        Company ID
    """
    ticker = company_data.get("ticker")
    if not ticker:
        raise ValueError("Company data must include ticker")
    
    # Update if exists, insert if not
    result = await db[COMPANIES_COLLECTION].update_one(
        {"ticker": ticker},
        {"$set": company_data},
        upsert=True
    )
    
    if result.upserted_id:
        return str(result.upserted_id)
    else:
        company = await get_company(db, ticker)
        return str(company["_id"]) if company else ""

# Document-related database operations
async def store_document_metadata(db: AsyncIOMotorDatabase, document_metadata: Dict[str, Any]) -> str:
    """
    Store document metadata in database
    
    Args:
        db: MongoDB database connection
        document_metadata: Document metadata to store
    
    Returns:
        Document ID
    """
    result = await db[DOCUMENTS_COLLECTION].insert_one(document_metadata)
    return str(result.inserted_id)

async def get_document_metadata(db: AsyncIOMotorDatabase, document_id: str) -> Optional[Dict[str, Any]]:
    """
    Get document metadata from database
    
    Args:
        db: MongoDB database connection
        document_id: Document ID
    
    Returns:
        Document metadata or None if not found
    """
    from bson.objectid import ObjectId
    
    try:
        return await db[DOCUMENTS_COLLECTION].find_one({"_id": ObjectId(document_id)})
    except:
        return await db[DOCUMENTS_COLLECTION].find_one({"id": document_id})

async def list_company_documents(db: AsyncIOMotorDatabase, ticker: str) -> List[Dict[str, Any]]:
    """
    List documents for a company
    
    Args:
        db: MongoDB database connection
        ticker: Company ticker symbol
    
    Returns:
        List of document metadata
    """
    cursor = db[DOCUMENTS_COLLECTION].find({"ticker": ticker})
    return await cursor.to_list(length=None)

# Research report-related database operations
async def store_research_report(db: AsyncIOMotorDatabase, report_data: Dict[str, Any]) -> str:
    """
    Store research report in database
    
    Args:
        db: MongoDB database connection
        report_data: Research report data
    
    Returns:
        Report ID
    """
    result = await db[RESEARCH_REPORTS_COLLECTION].insert_one(report_data)
    return str(result.inserted_id)

async def update_research_report(db: AsyncIOMotorDatabase, report_id: str, update_data: Dict[str, Any]) -> bool:
    """
    Update research report in database
    
    Args:
        db: MongoDB database connection
        report_id: Report ID
        update_data: Data to update
    
    Returns:
        Success status
    """
    from bson.objectid import ObjectId
    
    try:
        result = await db[RESEARCH_REPORTS_COLLECTION].update_one(
            {"_id": ObjectId(report_id)},
            {"$set": update_data}
        )
        return result.modified_count > 0
    except:
        result = await db[RESEARCH_REPORTS_COLLECTION].update_one(
            {"report_id": report_id},
            {"$set": update_data}
        )
        return result.modified_count > 0

async def get_research_report(db: AsyncIOMotorDatabase, report_id: str) -> Optional[Dict[str, Any]]:
    """
    Get research report from database
    
    Args:
        db: MongoDB database connection
        report_id: Report ID
    
    Returns:
        Research report or None if not found
    """
    from bson.objectid import ObjectId
    
    try:
        return await db[RESEARCH_REPORTS_COLLECTION].find_one({"_id": ObjectId(report_id)})
    except:
        return await db[RESEARCH_REPORTS_COLLECTION].find_one({"report_id": report_id})

async def list_company_research_reports(db: AsyncIOMotorDatabase, ticker: str) -> List[Dict[str, Any]]:
    """
    List research reports for a company
    
    Args:
        db: MongoDB database connection
        ticker: Company ticker symbol
    
    Returns:
        List of research reports
    """
    cursor = db[RESEARCH_REPORTS_COLLECTION].find({"ticker": ticker})
    return await cursor.to_list(length=None)

# Query history-related database operations
async def store_query(db: AsyncIOMotorDatabase, query_data: Dict[str, Any]) -> str:
    """
    Store query in database
    
    Args:
        db: MongoDB database connection
        query_data: Query data
    
    Returns:
        Query ID
    """
    result = await db[QUERY_HISTORY_COLLECTION].insert_one(query_data)
    return str(result.inserted_id)

async def list_recent_queries(db: AsyncIOMotorDatabase, limit: int = 10) -> List[Dict[str, Any]]:
    """
    List recent queries
    
    Args:
        db: MongoDB database connection
        limit: Maximum number of queries to return
    
    Returns:
        List of recent queries
    """
    cursor = db[QUERY_HISTORY_COLLECTION].find().sort("timestamp", -1).limit(limit)
    return await cursor.to_list(length=None)

# Ingestion history-related database operations
async def store_ingestion_record(db: AsyncIOMotorDatabase, ingestion_data: Dict[str, Any]) -> str:
    """
    Store ingestion record in database
    
    Args:
        db: MongoDB database connection
        ingestion_data: Ingestion data
    
    Returns:
        Ingestion record ID
    """
    result = await db[INGEST_HISTORY_COLLECTION].insert_one(ingestion_data)
    return str(result.inserted_id)

async def get_latest_ingestion(db: AsyncIOMotorDatabase, ticker: str) -> Optional[Dict[str, Any]]:
    """
    Get latest ingestion record for a company
    
    Args:
        db: MongoDB database connection
        ticker: Company ticker symbol
    
    Returns:
        Latest ingestion record or None if not found
    """
    cursor = db[INGEST_HISTORY_COLLECTION].find({"ticker": ticker}).sort("timestamp", -1).limit(1)
    results = await cursor.to_list(length=1)
    return results[0] if results else None

async def update_ingestion_status(db: AsyncIOMotorDatabase, ingestion_id: str, status: str, message: str = None) -> bool:
    """
    Update ingestion status
    
    Args:
        db: MongoDB database connection
        ingestion_id: Ingestion record ID
        status: New status
        message: Optional status message
    
    Returns:
        Success status
    """
    from bson.objectid import ObjectId
    
    update_data = {"status": status}
    if message:
        update_data["message"] = message
    
    try:
        result = await db[INGEST_HISTORY_COLLECTION].update_one(
            {"_id": ObjectId(ingestion_id)},
            {"$set": update_data}
        )
        return result.modified_count > 0
    except:
        result = await db[INGEST_HISTORY_COLLECTION].update_one(
            {"id": ingestion_id},
            {"$set": update_data}
        )
        return result.modified_count > 0