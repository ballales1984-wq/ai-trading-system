"""Minimal Vercel Python test."""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/api/v1/health")
async def health():
    return {"status": "ok"}

# Vercel requires a handler
def handler(event, context):
    """Vercel handler function."""
    return app(event, context)
