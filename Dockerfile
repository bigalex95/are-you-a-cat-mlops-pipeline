FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies (libgomp1 for TensorFlow) and clean up
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Install Python dependencies first for better caching
COPY requirements-docker.txt ./
RUN uv pip install --system --no-cache -r requirements-docker.txt

# Copy the rest of the application code
COPY . .

# Default command can be set by compose or overridden at runtime
