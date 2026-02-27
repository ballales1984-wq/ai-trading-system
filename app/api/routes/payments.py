"""
Stripe Payment Routes
===================
Handles Stripe checkout sessions and payment processing.
"""

import os
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel

from app.core.rbac import get_current_user
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


class CheckoutSessionRequest(BaseModel):
    """Request to create a Stripe checkout session."""
    price_id: Optional[str] = None
    success_url: Optional[str] = None
    cancel_url: Optional[str] = None


@router.post("/stripe/checkout-session")
async def create_checkout_session(
    request: CheckoutSessionRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Create a Stripe checkout session for lifetime purchase.
    """
    # Get Stripe keys from environment
    stripe_api_key = os.getenv("STRIPE_API_KEY") or os.getenv("STRIPE_SECRET_KEY")
    stripe_payment_link = os.getenv("VITE_STRIPE_PAYMENT_LINK")
    
    if not stripe_api_key and not stripe_payment_link:
        raise HTTPException(
            status_code=503,
            detail="Payment system not configured. Please contact support."
        )
    
    # If using payment link, redirect to it
    if stripe_payment_link:
        logger.info(f"Redirecting user {current_user.get('username')} to Stripe payment link")
        return RedirectResponse(url=stripe_payment_link, status_code=303)
    
    # Otherwise create checkout session using Stripe API
    try:
        import stripe
        stripe.api_key = stripe_api_key
        
        # Create checkout session
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency": "eur",
                    "product_data": {
                        "name": "Lifetime Access - AI Trading System",
                        "description": "One-time payment for lifetime access"
                    },
                    "unit_amount": 4999,  # €49.99
                },
                "quantity": 1
            }],
            mode="payment",
            success_url=request.success_url or "https://ai-trading-system-kappa.vercel.app/success",
            cancel_url=request.cancel_url or "https://ai-trading-system-kappa.vercel.app/cancel",
            metadata={
                "user_id": current_user.get("user_id"),
                "username": current_user.get("username")
            }
        )
        
        logger.info(f"Created checkout session for user {current_user.get('username')}")
        
        return {
            "session_id": checkout_session.id,
            "url": checkout_session.url
        }
        
    except Exception as e:
        logger.error(f"Error creating checkout session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    """
    Handle Stripe webhook events.
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    
    stripe_webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    
    if not stripe_webhook_secret:
        raise HTTPException(status_code=503, detail="Webhook not configured")
    
    try:
        import stripe
        stripe.api_key = os.getenv("STRIPE_API_KEY") or os.getenv("STRIPE_SECRET_KEY")
        
        event = stripe.Webhook.construct_event(
            payload, sig_header, stripe_webhook_secret
        )
        
        if event["type"] == "checkout.session.completed":
            session = event["data"]["object"]
            user_id = session.get("metadata", {}).get("user_id")
            logger.info(f"Payment completed for user {user_id}")
            # Update user subscription status here
            
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"status": "error", "detail": str(e)}
