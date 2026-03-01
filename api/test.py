"""
Minimal test API for Vercel debugging
"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/")
def root():
    return JSONResponse({"status": "ok", "message": "root"})

@app.get("/test")
def test():
    return JSONResponse({"status": "ok", "message": "test works"})
