# === Stage 1: Build ===
FROM python:3.11-slim AS builder

# Set environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VENV_PATH=/app/.venv

WORKDIR /app

# Install system dependencies needed for PyTorch/ONNX
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy models
COPY onnx_models/final_model_augv2_89/model.onnx /app/onnx_models/final_model_augv2_89/model.onnx
COPY backend/model/final_model_augv2_89 /app/backend/model/final_model_augv2_89

# Copy app source code
COPY . .

# === Stage 2: Runtime ===
FROM python:3.11-slim

WORKDIR /app

# Copy venv and app from builder
COPY --from=builder /app/.venv .venv
COPY --from=builder /app/onnx_models /app/onnx_models
COPY --from=builder /app/backend/model /app/backend/model
COPY --from=builder /app/main.py .  
 

# Ensure virtual environment is used
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8080

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "1", "-b", "0.0.0.0:$PORT", "main:app"]
