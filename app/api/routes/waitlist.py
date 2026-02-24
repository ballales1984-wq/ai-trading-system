"""
Waitlist API Routes
===================
Handle email signups for the waitlist.
"""

from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Request, status
from pydantic import BaseModel, EmailStr

from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

DATA_DIR = Path("data")
WAITLIST_FILE = DATA_DIR / "waitlist.json"
WAITLIST_LOCK = Lock()
RATE_LIMIT_SECONDS = 30
_last_submit_by_ip: Dict[str, datetime] = {}


class WaitlistEntry(BaseModel):
    """Waitlist entry schema."""

    email: EmailStr
    source: Optional[str] = "landing_page"

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "source": "landing_page",
            }
        }


class WaitlistResponse(BaseModel):
    """Response for waitlist signup."""

    success: bool
    message: str
    position: Optional[int] = None


def _load_waitlist() -> List[dict]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not WAITLIST_FILE.exists():
        WAITLIST_FILE.write_text("[]", encoding="utf-8")
        return []

    try:
        data = json.loads(WAITLIST_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        logger.exception("Failed to parse waitlist file. Resetting to empty list.")
        WAITLIST_FILE.write_text("[]", encoding="utf-8")
        return []


def _save_waitlist(entries: List[dict]) -> None:
    WAITLIST_FILE.write_text(json.dumps(entries, ensure_ascii=True, indent=2), encoding="utf-8")


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _enforce_rate_limit(client_ip: str) -> None:
    now = datetime.utcnow()
    last = _last_submit_by_ip.get(client_ip)
    if last and (now - last) < timedelta(seconds=RATE_LIMIT_SECONDS):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many submissions. Retry in {RATE_LIMIT_SECONDS} seconds.",
        )
    _last_submit_by_ip[client_ip] = now


@router.post("/waitlist", response_model=WaitlistResponse)
async def join_waitlist(entry: WaitlistEntry, request: Request):
    """
    Add email to waitlist.

    - **email**: Valid email address
    - **source**: Where the signup came from (optional)
    """
    normalized_email = entry.email.lower().strip()
    source = (entry.source or "landing_page").strip()[:100]
    client_ip = _get_client_ip(request)

    try:
        _enforce_rate_limit(client_ip)
        with WAITLIST_LOCK:
            entries = _load_waitlist()
            for existing in entries:
                if existing.get("email", "").lower() == normalized_email:
                    logger.info("waitlist_duplicate email=%s source=%s ip=%s", normalized_email, source, client_ip)
                    return WaitlistResponse(
                        success=True,
                        message="You're already on the waitlist!",
                        position=existing.get("position"),
                    )

            position = len(entries) + 1
            new_entry = {
                "email": normalized_email,
                "source": source,
                "created_at": datetime.utcnow().isoformat(),
                "position": position,
                "ip": client_ip,
            }
            entries.append(new_entry)
            _save_waitlist(entries)

        logger.info("waitlist_signup email=%s source=%s position=%s ip=%s", normalized_email, source, position, client_ip)
        return WaitlistResponse(
            success=True,
            message="Successfully joined the waitlist!",
            position=position,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("waitlist_signup_error email=%s ip=%s", normalized_email, client_ip)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to join waitlist. Please try again.",
        )


@router.get("/waitlist/count")
async def get_waitlist_count():
    """Get current waitlist count."""
    with WAITLIST_LOCK:
        entries = _load_waitlist()
    return {
        "count": len(entries),
        "last_signup": entries[-1]["created_at"] if entries else None,
    }


@router.get("/waitlist/export")
async def export_waitlist(x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key")):
    """Export waitlist as JSON. Requires admin key."""
    admin_key = os.getenv("ADMIN_SECRET_KEY")
    if not admin_key or not x_admin_key or x_admin_key != admin_key:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    with WAITLIST_LOCK:
        entries = _load_waitlist()
    return {"total": len(entries), "entries": entries}
