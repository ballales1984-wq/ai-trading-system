"""Script to start FastAPI server with uvicorn"""
import subprocess
import sys

if __name__ == "__main__":
    print("Starting FastAPI server on port 8000...")
    subprocess.run([sys.executable, "-m", "uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])

