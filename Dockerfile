FROM python:3.12-slim

# Environment settings for reliable logging and installs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UVICORN_TIMEOUT_KEEP_ALIVE=5

WORKDIR /app

# System deps (curl for health checks/debug), and build essentials if needed
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy project
COPY . .

# Default Fly internal port
ENV PORT=8080
EXPOSE 8080

# Start the FastAPI app (respect Fly's PORT if provided)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]


