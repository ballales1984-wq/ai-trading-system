# Vercel entry point
from app.main import app

# Vercel requires the handler to be named 'handler'
handler = app
