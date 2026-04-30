# ═══════════════════════════════════════════════════════════════
# AgriSense — Dockerfile
# ═══════════════════════════════════════════════════════════════
# Multi-stage build for optimized production image
# SDG 2: Zero Hunger — AI Crop Disease Detection System
# ═══════════════════════════════════════════════════════════════

# ── STAGE 1: Base Image
# Using Python 3.11 slim (smaller than full, larger than alpine)
# Slim has enough system libs for PyTorch + OpenCV
FROM python:3.11-slim AS base

# ── Set environment variables
# Prevents Python from writing .pyc files (saves disk space)
ENV PYTHONDONTWRITEBYTECODE=1
# Ensures Python output is not buffered (important for Docker logs)
ENV PYTHONUNBUFFERED=1
# Set Flask to production mode
ENV FLASK_ENV=production
ENV FLASK_DEBUG=false
# Default port
ENV PORT=5000

# ── Set working directory inside container
WORKDIR /app

# ── Install system dependencies
# libgomp1: required by PyTorch (OpenMP for parallel computation)
# libglib2.0-0: required by OpenCV
# We clean up apt cache to keep image small
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── STAGE 2: Install Python Dependencies
FROM base AS dependencies

# Copy only requirements.txt first (Docker layer caching optimization)
# If requirements.txt hasn't changed, this layer is cached → faster builds
COPY requirements.txt .

# Install Python packages
# --no-cache-dir: don't cache pip downloads (smaller image)
# --upgrade pip: ensure latest pip for compatibility
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── STAGE 3: Final Application Image
FROM dependencies AS final

# Copy application source code
# Ordered from least-frequently-changed to most-frequently-changed
# (for Docker layer caching efficiency)
COPY data/           ./data/
COPY model/          ./model/
COPY llm/            ./llm/
COPY templates/      ./templates/
COPY static/         ./static/
COPY app.py          .

# Create uploads directory (needs to exist at runtime)
RUN mkdir -p uploads

# Create a non-root user for security
# Running as root in production is a security risk
RUN useradd -m -u 1000 agrisense && chown -R agrisense:agrisense /app
USER agrisense

# ── Expose the Flask port
EXPOSE 5000

# ── Health check
# Docker will use this to determine if the container is healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# ── Start the application
# Using gunicorn for production (NOT flask dev server)
# -w 1: 1 worker (ML model is memory-heavy; increase if you have RAM)
# -b 0.0.0.0:5000: bind to all interfaces on port 5000
# --timeout 120: allow 120s for inference (model loading is slow first time)
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "120", "app:app"]
