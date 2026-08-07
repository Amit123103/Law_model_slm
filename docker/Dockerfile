# Multi-stage PyTorch deployment container for SLM ecosystem
FROM python:3.10-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "slm.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
