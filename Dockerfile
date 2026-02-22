# ============================================================
# AI Trading System - Render Dockerfile
# ============================================================
# Optimized for Render deployment

FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=$PORT
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-vercel.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir -r requirements-vercel.txt

# Copy frontend files and build
COPY frontend/ ./frontend/
RUN cd frontend && npm install && npm run build

# Copy application code
COPY api/ ./api/
COPY app/ ./app/
COPY .env.example .env

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port (Render sets PORT env var)
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:$PORT/api/v1/health || exit 1

# Run the application
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "$PORT"]
