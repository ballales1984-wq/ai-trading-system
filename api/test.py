"""
Minimal test API for Vercel debugging
"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/")
def root():
    return JSONResponse({"status": "ok"})

@app.get("/test")
def test():
    return JSONResponse({"message": "test works"})
