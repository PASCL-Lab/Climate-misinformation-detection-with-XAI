FROM python:3.11-slim as builder

# Install build deps only in builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install CPU-only versions of big libs
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       torch --index-url https://download.pytorch.org/whl/cpu \
       onnxruntime==1.17.0 \
    && pip install --no-cache-dir -r requirements.txt

# Final runtime image
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy only necessary code + models
COPY main.py .
COPY backend/model/final_model_augv2_89 ./models/
COPY onnx_models/final_model_augv2_89/model.onnx ./onnx_models/final_model_augv2_89/model.onnx

# Non-root user
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=30s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
