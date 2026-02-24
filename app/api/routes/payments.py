"""
Stripe payment routes.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import stripe
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, EmailStr

from app.core.config import settings
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


class CreateCheckoutRequest(BaseModel):
    email: Optional[EmailStr] = None
    price_id: Optional[str] = None
    quantity: int = 1


class CreateCheckoutResponse(BaseModel):
    checkout_url: str
    session_id: str


def _stripe_secret_key() -> str:
    # Backward compatibility: STRIPE_API_KEY used in existing env files.
    return settings.stripe_secret_key or os.getenv("STRIPE_API_KEY", "")


@router.post("/stripe/checkout-session", response_model=CreateCheckoutResponse)
async def create_checkout_session(payload: CreateCheckoutRequest) -> CreateCheckoutResponse:
    secret = _stripe_secret_key().strip()
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stripe is not configured yet (missing STRIPE_SECRET_KEY).",
        )

    price_id = (payload.price_id or settings.stripe_default_price_id).strip()
    if not price_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing Stripe price id. Configure STRIPE_DEFAULT_PRICE_ID or provide price_id.",
        )

    stripe.api_key = secret
    try:
        session = stripe.checkout.Session.create(
            mode="payment",
            line_items=[{"price": price_id, "quantity": max(1, payload.quantity)}],
            success_url=settings.stripe_success_url,
            cancel_url=settings.stripe_cancel_url,
            customer_email=payload.email or None,
            metadata={"source": "web_access_gate"},
        )
    except Exception as exc:
        logger.error("Stripe checkout session creation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to create Stripe checkout session.",
        ) from exc

    if not session.url:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Stripe checkout session has no URL.",
        )

    return CreateCheckoutResponse(checkout_url=session.url, session_id=session.id)


@router.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    signature = request.headers.get("stripe-signature", "")
    endpoint_secret = settings.stripe_webhook_secret.strip()
    secret = _stripe_secret_key().strip()

    if not secret:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Stripe not configured.")

    stripe.api_key = secret

    if endpoint_secret:
        try:
            event = stripe.Webhook.construct_event(payload=payload, sig_header=signature, secret=endpoint_secret)
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid Stripe webhook: {exc}")
    else:
        # Dev fallback if webhook secret is not configured.
        event = json.loads(payload.decode("utf-8"))

    event_type = event.get("type", "unknown")
    data_obj = event.get("data", {}).get("object", {})
    logger.info("Stripe webhook received: %s", event_type)

    if event_type == "checkout.session.completed":
        logger.info(
            "Stripe payment completed session_id=%s customer_email=%s",
            data_obj.get("id"),
            data_obj.get("customer_details", {}).get("email") or data_obj.get("customer_email"),
        )
        # TODO: Persist entitlement + license mapping in DB when schema is ready.

    return {"received": True, "type": event_type}
