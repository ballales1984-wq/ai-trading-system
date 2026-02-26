"""
Waitlist API Routes
====================
Handle email signups for the waitlist.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import Session

from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ==================== Models ====================

class WaitlistEntry(BaseModel):
    """Waitlist entry schema."""
    email: EmailStr
    source: Optional[str] = "landing_page"
    created_at: Optional[datetime] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "source": "landing_page"
            }
        }


class WaitlistResponse(BaseModel):
    """Response for waitlist signup."""
    success: bool
    message: str
    position: Optional[int] = None


# ==================== In-Memory Storage (for demo) ====================
# In production, this would be a database table

_waitlist: List[dict] = []


# ==================== Routes ====================

@router.post("/waitlist", response_model=WaitlistResponse)
async def join_waitlist(entry: WaitlistEntry):
    """
    Add email to waitlist.
    
    - **email**: Valid email address
    - **source**: Where the signup came from (optional)
    """
    try:
        # Check if email already exists
        for existing in _waitlist:
            if existing["email"] == entry.email:
                logger.info(f"Email already on waitlist: {entry.email}")
                return WaitlistResponse(
                    success=True,
                    message="You're already on the waitlist!",
                    position=existing["position"]
                )
        
        # Add new entry
        position = len(_waitlist) + 1
        new_entry = {
            "email": entry.email,
            "source": entry.source,
            "created_at": datetime.utcnow(),
            "position": position
        }
        _waitlist.append(new_entry)
        
        logger.info(f"New waitlist signup: {entry.email} (position {position})")
        
        return WaitlistResponse(
            success=True,
            message="Successfully joined the waitlist!",
            position=position
        )
        
    except Exception as e:
        logger.error(f"Waitlist signup error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to join waitlist. Please try again."
        )


@router.get("/waitlist/count")
async def get_waitlist_count():
    """Get the current waitlist count."""
    return {
        "count": len(_waitlist),
        "last_signup": _waitlist[-1]["created_at"] if _waitlist else None
    }


@router.get("/waitlist/export")
async def export_waitlist():
    """
    Export waitlist as JSON (for admin use).
    In production, this would require authentication.
    """
    return {
        "total": len(_waitlist),
        "entries": _waitlist
    }


# ==================== Database Model (for production) ====================
# Uncomment when using a real database

"""
from app.core.database import Base

class WaitlistModel(Base):
    __tablename__ = "waitlist"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    source = Column(String(100), default="landing_page")
    created_at = Column(DateTime, default=datetime.utcnow)
    notified = Column(Boolean, default=False)
    converted = Column(Boolean, default=False)
"""
