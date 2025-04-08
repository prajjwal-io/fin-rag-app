import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson.objectid import ObjectId
from ..services.auth_service import get_password_hash, User, UserInDB

logger = logging.getLogger(__name__)

# User collection name
USERS_COLLECTION = "users"

class UserService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
    
    async def get_user_by_username(self, username: str) -> Optional[UserInDB]:
        """
        Get user by username
        
        Args:
            username: Username
            
        Returns:
            User or None if not found
        """
        user_dict = await self.db[USERS_COLLECTION].find_one({"username": username})
        if user_dict:
            return UserInDB(**user_dict)
        return None
    
    async def get_user_by_id(self, user_id: str) -> Optional[UserInDB]:
        """
        Get user by ID
        
        Args:
            user_id: User ID
            
        Returns:
            User or None if not found
        """
        try:
            user_dict = await self.db[USERS_COLLECTION].find_one({"_id": ObjectId(user_id)})
            if user_dict:
                return UserInDB(**user_dict)
        except Exception as e:
            logger.error(f"Error getting user by ID: {str(e)}")
        return None
    
    async def create_user(
        self,
        username: str,
        password: str,
        email: Optional[str] = None,
        full_name: Optional[str] = None,
        role: str = "user"
    ) -> bool:
        """
        Create a new user
        
        Args:
            username: Username
            password: Plain password
            email: Optional email
            full_name: Optional full name
            role: User role (default: user)
            
        Returns:
            Success status
        """
        try:
            # Check if user already exists
            existing_user = await self.get_user_by_username(username)
            if existing_user:
                return False
            
            # Create new user
            user = {
                "username": username,
                "hashed_password": get_password_hash(password),
                "email": email,
                "full_name": full_name,
                "disabled": False,
                "created_at": datetime.now().isoformat(),
                "role": role
            }
            
            result = await self.db[USERS_COLLECTION].insert_one(user)
            return bool(result.inserted_id)
        
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            return False
    
    async def update_user(self, username: str, update_data: Dict[str, Any]) -> bool:
        """
        Update user data
        
        Args:
            username: Username
            update_data: Data to update
            
        Returns:
            Success status
        """
        try:
            # Don't allow updating username
            if "username" in update_data:
                del update_data["username"]
            
            # Hash password if provided
            if "password" in update_data:
                update_data["hashed_password"] = get_password_hash(update_data["password"])
                del update_data["password"]
            
            result = await self.db[USERS_COLLECTION].update_one(
                {"username": username},
                {"$set": update_data}
            )
            
            return result.modified_count > 0
        
        except Exception as e:
            logger.error(f"Error updating user: {str(e)}")
            return False
    
    async def delete_user(self, username: str) -> bool:
        """
        Delete user
        
        Args:
            username: Username
            
        Returns:
            Success status
        """
        try:
            result = await self.db[USERS_COLLECTION].delete_one({"username": username})
            return result.deleted_count > 0
        
        except Exception as e:
            logger.error(f"Error deleting user: {str(e)}")
            return False
    
    async def list_users(self, skip: int = 0, limit: int = 100) -> List[UserInDB]:
        """
        List users
        
        Args:
            skip: Number of users to skip
            limit: Maximum number of users to return
            
        Returns:
            List of users
        """
        try:
            cursor = self.db[USERS_COLLECTION].find().skip(skip).limit(limit)
            users = await cursor.to_list(length=limit)
            return [UserInDB(**user) for user in users]
        
        except Exception as e:
            logger.error(f"Error listing users: {str(e)}")
            return []
    
    async def count_users(self) -> int:
        """
        Count total users
        
        Returns:
            Number of users
        """
        try:
            return await self.db[USERS_COLLECTION].count_documents({})
        
        except Exception as e:
            logger.error(f"Error counting users: {str(e)}")
            return 0
    
    async def disable_user(self, username: str) -> bool:
        """
        Disable user
        
        Args:
            username: Username
            
        Returns:
            Success status
        """
        return await self.update_user(username, {"disabled": True})
    
    async def enable_user(self, username: str) -> bool:
        """
        Enable user
        
        Args:
            username: Username
            
        Returns:
            Success status
        """
        return await self.update_user(username, {"disabled": False})
    
    async def change_user_role(self, username: str, role: str) -> bool:
        """
        Change user role
        
        Args:
            username: Username
            role: New role
            
        Returns:
            Success status
        """
        return await self.update_user(username, {"role": role})