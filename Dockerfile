# ============================================================
# AI Trading System - Production Dockerfile
# ============================================================
# Multi-stage build for optimized production image
# 
# Build: docker build -t ai-trading-system:prod .
# Run:   docker run -p 8050:8050 ai-trading-system:prod

# ------------------------------------
# Stage 1: Builder
# ------------------------------------
FROM python:3.11-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir -r requirements.txt

# ------------------------------------
# Stage 2: Production
# ------------------------------------
FROM python:3.11-slim-bookworm AS production

# Create non-root user for security
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Create necessary directories with proper ownership
RUN mkdir -p /app/data /app/logs /app/cache /app/models && \
    chown -R appuser:appgroup /app

# Copy application code
COPY --chown=appuser:appgroup . .

# Switch to non-root user
USER appuser

# Expose ports
# Dashboard: 8050
# API Server: 8000
EXPOSE 8050 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8050/', timeout=5)" || exit 1

# Run the application
# Default to dashboard mode
CMD ["python", "main.py", "--mode", "dashboard", "--host", "0.0.0.0"]

# ------------------------------------
# Stage 3: Development (optional)
# ------------------------------------
FROM production AS development

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-cov ipython black flake8 mypy

# Switch back to root for development
USER root

# Install dev tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    && rm -rf /var/lib/apt/lists/*

USER appuser

# Default to test mode in dev
CMD ["python", "main.py", "--mode", "test"]
