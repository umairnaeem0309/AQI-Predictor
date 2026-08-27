# =============================================================================
# AQI Predictor — Dockerfile
# =============================================================================
# Multi-stage build for optimized image size
# Base image: Python 3.11 slim (required for Hopsworks compatibility)
# =============================================================================

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# --- Dependencies Stage ---
FROM base AS dependencies

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# Note: tensorflow-cpu is used by default; GPU support requires explicit configuration
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Application Stage ---
FROM dependencies AS app

# Copy source code
COPY src/ ./src/
COPY app/ ./app/
COPY pipelines/ ./pipelines/
COPY config.yaml .
COPY models/ ./models/

# Copy environment template (actual .env is mounted at runtime)
COPY .env.example .env

# Copy processed data for feature adapter
COPY data/processed/ ./data/processed/

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: run FastAPI via uvicorn
CMD ["python", "-m", "uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
