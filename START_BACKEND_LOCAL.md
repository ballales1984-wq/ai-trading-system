# Starting the Backend Locally

This guide explains how to run the AI Trading System backend locally for development and testing with the Vercel-hosted frontend.

## Prerequisites

- Python 3.11+
- [ngrok](https://ngrok.com/download) (for exposing local backend to Vercel)

## Quick Start

### Option 1: Using the Batch Script (Windows)

Run the provided batch script to start both the backend and ngrok:

```batch
start_backend_with_ngrok.bat
```

This will:
1. Install Python dependencies
2. Start the FastAPI backend on port 8000
3. Start ngrok to expose the backend publicly

### Option 2: Manual Setup

1. **Start the backend:**
   ```bash
   pip install -r requirements.txt
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

2. **Start ngrok (in a separate terminal):**
   ```bash
   ngrok http 8000
   ```

## Configuring Vercel Frontend

Once ngrok is running, you'll see a public HTTPS URL (e.g., `https://abc123.ngrok.io`).

1. Go to your Vercel project settings
2. Navigate to **Environment Variables**
3. Add or update `VITE_API_BASE_URL`:
   ```
   VITE_API_BASE_URL=https://your-ngrok-url.ngrok.io/api/v1
   ```
4. Redeploy the frontend for changes to take effect

## API Endpoints

The backend provides the following main endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /api/v1/portfolio/summary` | Portfolio summary |
| `GET /api/v1/market/prices` | Market prices |
| `GET /api/v1/orders` | List orders |
| `POST /api/v1/orders` | Create order |
| `GET /api/v1/strategy/performance` | Strategy performance |
| `GET /api/v1/risk/metrics` | Risk metrics |
| `POST /api/v1/waitlist` | Join waitlist |

## Troubleshooting

### Backend not responding
- Check if port 8000 is already in use
- Verify all dependencies are installed

### CORS errors
- Ensure the backend CORS settings include your frontend URL
- Check that `VITE_API_BASE_URL` is correctly set in Vercel

### ngrok connection issues
- Verify ngrok is installed and in your PATH
- Check your ngrok account limits if using free tier

## Development vs Production

| Setting | Development | Production |
|---------|-------------|------------|
| CORS Origins | `["*"]` | Specific domains |
| Debug Mode | `True` | `False` |
| API URL | ngrok URL | Production API URL |
