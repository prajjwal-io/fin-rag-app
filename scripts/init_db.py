import asyncio
import logging
from motor.motor_asyncio import AsyncIOMotorClient
import sys
import os

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config.settings import settings
from app.services.auth_service import create_user
from app.db.mongodb import COMPANIES_COLLECTION, DOCUMENTS_COLLECTION
from app.db.mongodb import RESEARCH_REPORTS_COLLECTION, QUERY_HISTORY_COLLECTION, INGEST_HISTORY_COLLECTION
from app.db.mongodb import close_db_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

async def init_db():
    # Connect to MongoDB
    logger.info(f"Connecting to MongoDB at {settings.MONGODB_URI}")
    client = AsyncIOMotorClient(settings.MONGODB_URI)
    db = client[settings.MONGODB_DB_NAME]
    
    # Create collections with indexes
    logger.info("Creating collections and indexes...")
    
    # Companies collection
    await db.create_collection(COMPANIES_COLLECTION)
    await db[COMPANIES_COLLECTION].create_index("ticker", unique=True)
    
    # Documents collection
    await db.create_collection(DOCUMENTS_COLLECTION)
    await db[DOCUMENTS_COLLECTION].create_index("ticker")
    await db[DOCUMENTS_COLLECTION].create_index("content_type")
    await db[DOCUMENTS_COLLECTION].create_index([("ticker", 1), ("content_type", 1)])
    
    # Research reports collection
    await db.create_collection(RESEARCH_REPORTS_COLLECTION)
    await db[RESEARCH_REPORTS_COLLECTION].create_index("report_id", unique=True)
    await db[RESEARCH_REPORTS_COLLECTION].create_index("ticker")
    
    # Query history collection
    await db.create_collection(QUERY_HISTORY_COLLECTION)
    await db[QUERY_HISTORY_COLLECTION].create_index("timestamp")
    await db[QUERY_HISTORY_COLLECTION].create_index("ticker")
    
    # Ingestion history collection
    await db.create_collection(INGEST_HISTORY_COLLECTION)
    await db[INGEST_HISTORY_COLLECTION].create_index("ticker")
    await db[INGEST_HISTORY_COLLECTION].create_index("timestamp")
    
    # Users collection
    await db.create_collection("users")
    await db["users"].create_index("username", unique=True)
    if "email" in db["users"].list_indexes():
        await db["users"].create_index("email", unique=True, sparse=True)
    
    # Create default admin user if not exists
    admin_username = os.environ.get("ADMIN_USERNAME", "admin")
    admin_password = os.environ.get("ADMIN_PASSWORD", "adminpassword")
    
    existing_admin = await db["users"].find_one({"username": admin_username})
    if not existing_admin:
        logger.info(f"Creating default admin user: {admin_username}")
        success = await create_user(
            db=db,
            username=admin_username,
            password=admin_password,
            email="admin@example.com",
            role="admin"
        )
        
        if success:
            logger.info("Admin user created successfully")
        else:
            logger.error("Failed to create admin user")
    
    logger.info("Database initialization completed")
    
    # Close connection
    client.close()

if __name__ == "__main__":
    logger.info("Starting database initialization")
    asyncio.run(init_db())
    logger.info("Database initialization completed")
