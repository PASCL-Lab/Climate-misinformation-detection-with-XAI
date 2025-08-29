# ===== Stage 1: Builder =====
FROM python:3.10-slim-bullseye AS builder

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install packages to /install
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ===== Stage 2: Final image =====
FROM python:3.10-slim-bullseye

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy only essential app files
COPY main.py .
COPY onnx_models/final_model_augv2_89/model.onnx ./onnx_models/final_model_augv2_89/model.onnx
COPY backend/model/final_model_augv2_89 ./backend/model/final_model_augv2_89

# Clean __pycache__ to save space
RUN find /usr/local/lib/python3.10/site-packages -name "__pycache__" -type d -exec rm -rf {} +

# Expose Fly.io port
EXPOSE 8080

# Start FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
