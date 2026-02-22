# Vercel entry point for FastAPI
from app.main import app

# Vercel Python runtime expects an ASGI app named 'handler'
handler = app
